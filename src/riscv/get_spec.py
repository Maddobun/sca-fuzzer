#!/usr/bin/env python3

import copy
import math
import re
import os
import glob
import json
import subprocess
from typing import List
from xml.etree import ElementTree as ET


class OperandSpec:
    name: str
    values: List[str]
    type_: str
    width: int
    comment: str
    src: bool = False
    dest: bool = False

    def to_json(self) -> str:
        return json.dumps(self, default=vars)


class InstructionSpec:
    name: str
    category: str = ""
    control_flow: bool = False
    operands: List[OperandSpec]
    implicit_operands: List[OperandSpec]
    datasize: int = 0

    def __init__(self) -> None:
        self.operands = []
        self.implicit_operands = []

    def __str__(self) -> str:
        return (
            f"{self.name} {self.control_flow} {self.category} "
            f"{len(self.operands)} {len(self.implicit_operands)}"
        )

    def to_json(self) -> str:
        s = "{"
        s += f'"name": "{self.name}", "category": "{self.category}", '
        s += f'"control_flow": {str(self.control_flow).lower()},\n'
        s += '  "operands": [\n    '
        s += ",\n    ".join([o.to_json() for o in self.operands])
        s += "\n  ],\n"
        if self.implicit_operands:
            s += '  "implicit_operands": [\n    '
            s += ",\n    ".join([o.to_json() for o in self.implicit_operands])
            s += "\n  ]"
        else:
            s += '  "implicit_operands": []'
        s += "\n}"
        return s


class ParseFailed(Exception):
    pass


GPRs = ["x9", "x10", "x11", "x12", "x13", "x14"]
FPGPRs = ["f8", "f9", "f10", "f11", "f12", "f13", "f14", "f15"]
unordered = [
    "FCVT.WU.S",
    "FCVT.WU.D",
    "FCVT.S.W",
    "FCVT.S.WU",
    "FCVT.S.D",
    "FCVT.D.S",
    "FCVT.D.W",
    "FCVT.D.WU",
    "FCVT.W.D",
    "FCVT.W.S",
    "FMV.W.X",
    "FMV.X.W",
    "FCLASS.S",
    "FCLASS.D",
]
ordered = ["FLD", "FLW", "FSW", "FSD", "C.FSD", "C.FLD", "C.BNEZ", "C.BEQZ"]


class RISCVTransformer:
    tree: ET.ElementTree
    instructions: List[InstructionSpec]
    current_spec: InstructionSpec

    def __init__(self) -> None:
        self.instructions = []

    def load_files(self, files: List[str]):
        # get the data from all files
        tree = root = ET.Element("root")
        for filename in files:
            data = ET.parse(filename).getroot()
            root.append(data)
        if not tree:
            print("No input. Exiting")
            exit(1)
        self.tree = tree

    def parse_tree(self):
        uses_pc: bool
        category: str
        make_me_labelled: bool
        make_me_with_no_offset: bool
        make_me_im_atomic: bool
        for instruction_group in self.tree.iter("instruction_file"):
            for instruction_node in instruction_group:
                self.current_spec = InstructionSpec()
                self.current_spec.name = instruction_node.attrib["name"]
                if "class" in instruction_node.attrib:
                    self.current_spec.category = instruction_node.attrib["class"]
                else:
                    self.current_spec.category = instruction_node.attrib["group"]
                make_me_labelled = False
                make_me_with_no_offset = False
                # not all instructions have an extension, somehow
                if "extension" in instruction_node.attrib:
                    if instruction_node.attrib["extension"] in [
                        "RV32I",
                        "RV32M",
                        "RV32F",
                        "RV32D",
                        "RV32Q",
                        "RV32C",
                        "RV32A",
                        "Zifencei",
                    ]:
                        self.current_spec.datasize = 32
                    elif instruction_node.attrib["extension"] in ["RV64D", "RV64C"]:
                        self.current_spec.datasize = 64
                    else:  # vector, system instructions
                        continue
                else:
                    continue  # extensionless instructions

                if "class" in instruction_node.attrib:
                    category = instruction_node.attrib["class"]
                else:
                    category = ""
                if category == "System":
                    continue  # No system instructions
                # implicit PCs
                if category == "BranchInstruction":
                    # Will need to make a LABEL version of this instruction, if it's not JALR or a C-extension version of JALR
                    make_me_labelled = (
                        True
                        if self.current_spec.name not in ["C.JR", "C.JALR", "JALR"]
                        else False
                    )
                    self.current_spec.control_flow = True
                    # EXCEPT for C.JALR, C.JR and JALR,
                    # all branch instructions are PC-based
                    uses_pc = True
                    for operand_node in instruction_node.findall("O"):
                        # branch instructions based on rs1 instead
                        if operand_node.attrib["name"] in [
                            "Branch-rs1",
                            "Branch-rs1-simm12",
                        ]:
                            uses_pc = False
                    if uses_pc:
                        op_pc = OperandSpec()
                        op_pc.name = "PC"
                        op_pc.src = True
                        op_pc.dest = False
                        op_pc.type_ = "REG"
                        op_pc.width = 64
                        op_pc.values = ["PC"]
                        self.current_spec.implicit_operands.append(op_pc)
                else:
                    self.current_spec.control_flow = False

                # Non-PC operand parsing
                for operand_node in instruction_node.findall("O"):
                    if operand_node.attrib["name"] == "const_bits":
                        continue  # We don't have time for "constant bits"

                    # Nested operands, found in branches and load/stores
                    if operand_node.attrib["type"] in ["Branch", "LoadStore"]:
                        # if we've found a CONDitional branch, get it's COND operand
                        if (
                            operand_node.attrib["type"] == "Branch"
                            and "condition" in operand_node.attrib
                        ):
                            cond_op = OperandSpec()
                            cond_op.name = "COND"
                            cond_op.src = False
                            cond_op.dest = False
                            cond_op.type_ = "COND"
                            cond_op.comment = "Branch condition to be met"
                            cond_op.width = 0
                            cond_op.values = [operand_node.attrib["condition"]]
                            self.current_spec.implicit_operands.append(cond_op)
                        is_implied = False
                        for operand_subnode in operand_node.findall("O"):
                            # if a register is implied, its implicit, and if it's got an offset, that's implicit too
                            is_implied = (
                                True
                                if (
                                    "class" in operand_subnode.attrib
                                    and operand_subnode.attrib["class"]
                                    == "ImpliedRegisterOperand"
                                )
                                or is_implied
                                else False
                            )
                            op = OperandSpec()
                            op.name = operand_subnode.attrib["name"]
                            op.src = True
                            op.dest = False
                            op.type_ = operand_subnode.attrib["type"]
                            if operand_node.attrib["type"] == "Branch":
                                op.comment = "Labelable"
                            if op.type_ == "GPR":
                                op.type_ = "REG"
                                if operand_node.attrib["type"] == "LoadStore":
                                    op.type_ = "MEM"
                            if op.type_ == "Immediate":
                                op.type_ = "IMM"
                                shamt = False
                                ld_st = False
                                scale = 1
                                low = 0
                                if (
                                    "class" in operand_subnode.attrib
                                    and operand_subnode.attrib["class"].find("Exclude")
                                    != -1
                                ):
                                    low = 1
                                if operand_node.attrib["type"] == "LoadStore":
                                    ld_st = True
                                    is_implied = (
                                        True
                                        if "extension" in instruction_node.attrib
                                        and instruction_node.attrib["extension"]
                                        == "RV64C"
                                        else is_implied
                                    )
                                    make_me_with_no_offset = (
                                        True if not is_implied else False
                                    )
                                    if "offset-scale" in operand_node.attrib:
                                        scale = pow(
                                            2,
                                            int(
                                                re.search(
                                                    r"\d+",
                                                    operand_node.attrib["offset-scale"],
                                                ).group()
                                            ),
                                        )
                                        if scale != 1:
                                            op.comment = (
                                                "offset of MEM (always >= 0 (sandbox)) to be scaled by "
                                                + str(scale)
                                            )
                                        else:
                                            op.comment = (
                                                "offset of MEM (always >= 0 (sandbox))"
                                            )
                                    else:
                                        op.comment = (
                                            "offset of MEM (always >= 0 (sandbox))"
                                        )
                                try:
                                    op.width = int(
                                        re.search(
                                            r"\d+", operand_subnode.attrib["name"]
                                        ).group()
                                    )
                                except AttributeError:
                                    op.width = int(
                                        math.log(self.current_spec.datasize, 2)
                                    )
                                    op_values = [1, (pow(2, op.width) - 1)]
                                    shamt = True
                                if (
                                    not shamt
                                    and operand_node.attrib["name"].find("simm") != -1
                                ):  # signed imm
                                    # though these may be signed imms, if they are found next to a MEM, we'll just make them always >= 0 (~sandboxing)
                                    op_values = (
                                        [
                                            (0 - pow(2, op.width - 1)),
                                            (pow(2, op.width - 1) - 1),
                                        ]
                                        if not ld_st
                                        else [0, pow(2, op.width - 1) - 1]
                                    )
                                elif not shamt:  # unsigned imm
                                    op_values = [low, (pow(2, op.width) - 1)]
                                op.values = [
                                    "[" + "-".join(str(i) for i in op_values) + "]"
                                ]
                            else:
                                op.width = self.current_spec.datasize
                                if is_implied:
                                    op.values = [operand_subnode.attrib["name"]]
                                else:
                                    op.values = (
                                        [operand_subnode.attrib["choices"]]
                                        if operand_subnode.attrib["choices"]
                                        != "Prime GPRs"
                                        else GPRs
                                    )
                            if is_implied:
                                self.current_spec.implicit_operands.append(op)
                            else:
                                self.current_spec.operands.append(op)
                        continue

                    # "Regular" operands
                    else:
                        is_implied = (
                            True
                            if "class" in operand_node.attrib
                            and operand_node.attrib["class"] == "ImpliedRegisterOperand"
                            else False
                        )
                        exp_op = OperandSpec()
                        exp_op.name = operand_node.attrib["name"]
                        if "access" in operand_node.attrib:
                            if operand_node.attrib["access"] == "ReadWrite":
                                exp_op.src = True
                                exp_op.dest = True
                            else:
                                exp_op.src = (
                                    True
                                    if operand_node.attrib["access"] == "Read"
                                    else False
                                )
                                exp_op.dest = (
                                    True
                                    if operand_node.attrib["access"] == "Write"
                                    else False
                                )
                        else:
                            exp_op.src = True
                            exp_op.dest = False
                        exp_op.type_ = operand_node.attrib["type"]
                        if exp_op.type_ == "Immediate":
                            exp_op.type_ = "IMM"
                            if operand_node.attrib["name"] in ["aq", "rl"]:
                                exp_op.comment = operand_node.attrib["name"]
                                exp_op.width = 1
                                exp_op.values = ["[0-1]"]
                                self.current_spec.implicit_operands.append(exp_op)
                                continue
                            shamt = False
                            if "choices" in operand_node.attrib:
                                # it's a FENCE op, and it's implicit
                                exp_op.width = 4
                                if operand_node.attrib["name"] == "fm":
                                    exp_op.values = [
                                        "[0-0]"
                                    ]  # FIXME fm should only be 0 or 8, though the generator.py has issues with this
                                    # using [0-0] range since 0 is the default value
                                    # (add a patch, maybe? change generator.py, perhaps?)
                                    # in any case, 0 enables normal fence mode, so let's stick with that
                                    exp_op.comment = "fence mode, 0000 or 1000"
                                else:
                                    # exp_op.values = ["[0-15]"]
                                    exp_op.values = [
                                        "[3-3]"
                                    ]  # we just always want it to be 3 (Read and Write)
                                    exp_op.comment = (
                                        operand_node.attrib["name"] + " : " + "option"
                                    )
                                self.current_spec.implicit_operands.append(exp_op)
                                continue
                            else:
                                low = 0
                                if (
                                    "exclude" in operand_node.attrib
                                ):  # only exclude case is excluding 0, so low is set to 1
                                    low = 1
                                try:
                                    exp_op.width = int(
                                        re.search(
                                            r"\d+", operand_node.attrib["name"]
                                        ).group()
                                    )
                                except AttributeError:
                                    exp_op.width = int(
                                        math.log(self.current_spec.datasize, 2)
                                    )
                                    op_values = [1, pow(2, exp_op.width) - 1]
                                    shamt = True
                                if (
                                    not shamt
                                    and operand_node.attrib["name"].find("simm") != -1
                                ):  # signed imm
                                    op_values = (
                                        [
                                            0 - pow(2, exp_op.width - 1),
                                            pow(2, exp_op.width - 1) - 1,
                                        ]
                                        if self.current_spec.name
                                        not in ["LUI", "AUIPC"]
                                        else [low, pow(2, exp_op.width) - 1]
                                    )  # AUIPC and LUI IMMs are described as signed, yet they aren't
                                elif not shamt:  # unsigned imm
                                    if self.current_spec.name in [
                                        "C.LI",
                                        "C.LUI",
                                        "C.ANDI",
                                        "C.ADDI",
                                    ]:  # special behavior
                                        exp_op.width -= 1
                                    op_values = [low, pow(2, exp_op.width) - 1]
                                exp_op.values = [
                                    "[" + "-".join(str(i) for i in op_values) + "]"
                                ]
                                if self.current_spec.name == "FENCE.I":
                                    # special case, FENCE.Is should have imm values set to 0
                                    exp_op.values = ["[0-0]"]
                        elif exp_op.type_ == "Choices":
                            # FIXME don't know how to deal with these - the generator can't deal with
                            # imm value sets, only imm value *ranges*
                            if operand_node.attrib["name"] == "rm":
                                exp_op.comment = "rounding mode"
                                exp_op.width = 3
                                # exp_op.values = ["[0,1,2,3,4,7]"]
                                exp_op.values = ["[0-4]"]
                                exp_op.type_ = "IMM"
                                self.current_spec.implicit_operands.append(exp_op)
                            continue
                        elif exp_op.type_ == "GPR":
                            exp_op.type_ = "REG"
                            exp_op.width = self.current_spec.datasize
                            if self.current_spec.name in ["FENCE", "FENCE.I"]:
                                # FENCEs: we are only interested in the read/write FENCE (similar behavior to MFENCE in x86)
                                exp_op.values = [
                                    "rw"
                                ]  # there are actually more choices
                                exp_op.comment = (
                                    "FENCE - register values should be 'rw' "
                                )
                            elif "choices" in operand_node.attrib:
                                if operand_node.attrib["choices"] in [
                                    "GPRs",
                                    "GPRs not x0, x2",
                                    "Prime GPRs",
                                ]:  # the GPRs we use are Prime GPRs
                                    exp_op.values = GPRs
                                else:
                                    exp_op.values = [operand_node.attrib["choices"]]
                            else:
                                exp_op.values = [operand_node.attrib["name"]]
                        elif exp_op.type_ == "FPR":
                            exp_op.type_ = "REG"
                            exp_op.width = int(
                                re.search(
                                    r"\d+", operand_node.attrib["choices"]
                                ).group()
                            )
                            # exp_op.values = [operand_node.attrib["choices"]]
                            exp_op.values = FPGPRs
                    if is_implied:
                        if "x0" not in exp_op.values:
                            self.current_spec.implicit_operands.append(
                                exp_op
                            )  # turned out to be implicit; we don't want x0 as implicit operand
                    else:
                        self.current_spec.operands.append(exp_op)

                # operand reordering (they aren't ordered in the .xmls, with the exception of A-extension instructions)
                if instruction_node.attrib["extension"] not in ["RV32A", "RV64A"]:
                    ordered_operands: List[OperandSpec] = []
                    if self.current_spec.name in unordered:
                        # On several cases with RV32/64D extensions, when there are two opperands, these are not ordered
                        # properly in the .xmls, and the asm fields in the .xmls are wrong, too.
                        ordered_operands.append(self.current_spec.operands[1])
                        ordered_operands.append(self.current_spec.operands[0])
                    elif self.current_spec.name not in ordered:
                        for operand_node in instruction_node.findall("asm"):
                            if self.current_spec.operands:
                                i = 1
                                operand_p = operand_node.attrib["op1"]
                                operand_ordered: OperandSpec()
                                while operand_p != None:
                                    for operand_unordered in self.current_spec.operands:
                                        if operand_unordered.name == operand_p:
                                            operand_ordered = operand_unordered
                                            self.current_spec.operands.remove(
                                                operand_unordered
                                            )
                                            break
                                    ordered_operands.append(operand_ordered)
                                    i += 1
                                    try:
                                        operand_p = operand_node.attrib["op" + str(i)]
                                    except KeyError:
                                        break
                    if ordered_operands.__len__() != 0:
                        self.current_spec.operands = ordered_operands
                else:
                    make_me_im_atomic = True
                # Operands / Operand sets which could be replaced with a LABEL
                if make_me_labelled:
                    copy_spec = copy.deepcopy(self.current_spec)
                    copy_ops = []
                    copy_ops_index = (
                        -1
                    )  # there should only be one Labelable, so no List of indices
                    for copy_op in copy_spec.operands:
                        try:
                            if copy_op.comment == "Labelable":
                                copy_ops.append(copy_op)
                        except AttributeError:
                            continue
                    for copy_op in copy_ops:
                        copy_ops_index = copy_spec.operands.index(copy_op)
                        copy_spec.operands.remove(copy_op)
                    label_op = OperandSpec()
                    label_op.name = "LABEL"
                    label_op.src = True
                    label_op.dest = False
                    label_op.type_ = "LABEL"
                    label_op.width = 0
                    label_op.values = []
                    copy_spec.operands.insert(copy_ops_index, label_op)
                    self.instructions.append(copy_spec)
                self.instructions.append(self.current_spec)
                # IMMs which are offsets of MEM operands which could be discarded when parsing already generated tests
                # these specs are added later rather than sooner since we don't want the generator to generate programs
                # using these
                if make_me_with_no_offset:
                    copy_spec = copy.deepcopy(self.current_spec)
                    copy_ops = []
                    copy_spec.category = (
                        "FORMAT"  # Instruction specs used only to read instructions
                    )
                    copy_ops_index = -1  # there should only be one offset
                    for copy_op in copy_spec.operands:
                        try:
                            if copy_op.comment.find("offset") != -1:
                                copy_ops.append(copy_op)
                        except AttributeError:
                            continue
                    for copy_op in copy_ops:
                        copy_ops_index = copy_spec.operands.index(copy_op)
                        copy_spec.operands.remove(copy_op)
                    for copy_op in copy_spec.operands:
                        if copy_op.type_ == "MEM":
                            copy_op.comment = "has an offset"
                    self.instructions.append(copy_spec)
                # Atomic instructions appear in several formats, having the aq and rl bits set their memory access order / visibility
                # e.g., AMOADD.W.AQ, AMOADD.W.RL, AMOADD.W.AQ.RL instead of just AMOADD.W
                # TODO: Should these formats be included?
                # if make_me_im_atomic:

        # adding RET pseudoinstruction
        self.current_spec = InstructionSpec()
        self.current_spec.name = "RET"
        self.current_spec.category = "General"
        self.current_spec.datasize = 32
        self.current_spec.control_flow = True
        x0_op = OperandSpec()
        x0_op.src = False
        x0_op.dest = True
        x0_op.type_ = "REG"
        x0_op.width = 32
        x0_op.values = ["x0"]
        x1_op = OperandSpec()
        x1_op.src = True
        x1_op.dest = False
        x1_op.type_ = "REG"
        x1_op.width = 32
        x1_op.values = ["x1"]
        zero_op = OperandSpec()
        zero_op.src = True
        zero_op.dest = False
        zero_op.type_ = "IMM"
        zero_op.width = 12
        zero_op.values = ["[0-0]"]
        self.current_spec.implicit_operands.append(x0_op)
        self.current_spec.implicit_operands.append(x1_op)
        self.current_spec.implicit_operands.append(zero_op)
        self.instructions.append(self.current_spec)

    def save(self, filename: str):
        json_str = "[\n" + ",\n".join([i.to_json() for i in self.instructions]) + "\n]"
        with open(filename, "w+") as f:
            f.write(json_str)


class Downloader:
    def __init__(self, extensions: List[str], out_file: str) -> None:
        self.extentions = extensions
        self.out_file = out_file

    def run(self):
        subprocess.run("mkdir instructions", shell=True, check=True)
        subprocess.run(
            "wget "
            "https://raw.githubusercontent.com/openhwgroup/force-riscv/master/riscv/arch_data/instr/g_instructions.xml",
            shell=True,
            check=True,
        )
        # for now, don't include vector instructions
        # subprocess.run(
        #    "wget "
        #    "https://raw.githubusercontent.com/openhwgroup/force-riscv/master/riscv/arch_data/instr/v_instructions.xml", shell=True, check=True)
        subprocess.run(
            "wget "
            "https://raw.githubusercontent.com/openhwgroup/force-riscv/master/riscv/arch_data/instr/zfh_instructions.xml",
            shell=True,
            check=True,
        )
        subprocess.run(
            "wget "
            "https://raw.githubusercontent.com/openhwgroup/force-riscv/master/riscv/arch_data/instr/c_instructions.xml",
            shell=True,
            check=True,
        )
        # All instruction descs -> instructions dir
        subprocess.run("mv *.xml instructions", shell=True, check=True)
        files = glob.glob("instructions/*.xml")

        try:
            transformer = RISCVTransformer()
            transformer.load_files(files)
            transformer.parse_tree()
            print(
                f"Produced base.json with {len(transformer.instructions)} instructions"
            )
            transformer.save("base.json")
        finally:
            subprocess.run("rm -r instructions", shell=True, check=True)
