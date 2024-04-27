"""
File: RISC-V implementation of the test generator


"""

import abc
import math
import random
import re
from typing import List, Dict, Tuple
from subprocess import run, CalledProcessError

from isa_loader import InstructionSet
from interfaces import (
    PageTableModifier,
    TestCase,
    Operand,
    RegisterOperand,
    MemoryOperand,
    ImmediateOperand,
    LabelOperand,
    OT,
    Instruction,
    BasicBlock,
    InstructionSpec,
)
from generator import (
    ConfigurableGenerator,
    RandomGenerator,
    Pass,
    parser_assert,
    Printer,
    GeneratorException,
    AsmParserException,
)
from riscv.riscv_target_desc import RISCVTargetDesc
from config import CONF
from service import LOGGER


class RISCVGenerator(ConfigurableGenerator, abc.ABC):

    def __init__(self, instruction_set: InstructionSet, seed: int):
        super(RISCVGenerator, self).__init__(instruction_set, seed)
        self.target_desc = RISCVTargetDesc()
        self.printer = RISCVPrinter()
        self.passes = [
            RISCVZeroAndMemPass(self.target_desc),
            RISCVSandboxPass(self.target_desc),
        ]
        # select PTE bits that could be set
        self.pte_bit_choices: List[Tuple[int, bool]] = []
        if "assist-accessed" in CONF.permitted_faults:
            self.pte_bit_choices.append(self.target_desc.pte_bits["ACCESSED"])
        if "assist-dirty" in CONF.permitted_faults:
            self.pte_bit_choices.append(self.target_desc.pte_bits["DIRTY"])
        if "PF-present" in CONF.permitted_faults:
            self.pte_bit_choices.append(self.target_desc.pte_bits["PRESENT"])
        if "PF-writable" in CONF.permitted_faults:
            self.pte_bit_choices.append(self.target_desc.pte_bits["RW"])
        if "PF-smap" in CONF.permitted_faults:
            self.pte_bit_choices.append(self.target_desc.pte_bits["USER"])

    @staticmethod
    def assemble(asm_file: str, bin_file: str) -> None:
        """Assemble the test case into a stripped binary"""

        def pretty_error_msg(error_msg):
            with open(asm_file, "r") as f:
                lines = f.read().split("\n")

            msg = "Error appeared while assembling the test case:\n"
            for line in error_msg.split("\n"):
                line = line.removeprefix(asm_file + ":")
                line_num_str = re.search(r"(\d+):", line)
                if not line_num_str:
                    msg += line
                else:
                    parsed = lines[int(line_num_str.group(1)) - 1]
                    msg += f"\n  Line {line}\n    (the line was parsed as {parsed})"
            return msg

        try:
            out = run(
                f"riscv64-linux-gnu-as {asm_file} -o {bin_file} -march=rv64imafdqc",
                shell=True,
                check=True,
                capture_output=True,
            )
        except CalledProcessError as e:
            error_msg = e.stderr.decode()
            if "Assembler messages:" not in error_msg:
                print(error_msg)
                raise e
            LOGGER.error(pretty_error_msg(error_msg))

        output = out.stderr.decode()
        if "Assembler messages:" in output:
            LOGGER.warning("generator", pretty_error_msg(output))

        run(
            f"riscv64-linux-gnu-strip --remove-section=.note.gnu.property {bin_file}",
            shell=True,
            check=True,
        )
        run(
            f"riscv64-linux-gnu-objcopy {bin_file} -O binary {bin_file}",
            shell=True,
            check=True,
        )

    def map_addresses(self, test_case: TestCase, bin_file: str) -> None:
        # get a list of relative instruction addresses
        dump = run(
            f"riscv64-linux-gnu-objdump -D -b binary -m riscv:rv64 {bin_file} "
            "| awk '/ [0-9a-f]+:/{print $1}'",
            shell=True,
            check=True,
            capture_output=True,
        )
        address_list = [
            int(addr[:-1], 16) for addr in dump.stdout.decode().split("\n") if addr
        ]

        # connect them with instructions in the test case
        address_map: Dict[int, Instruction] = {}
        counter = test_case.num_prologue_instructions
        for func in test_case.functions:
            for bb in func:
                for inst in list(bb) + bb.terminators:
                    address = address_list[counter]
                    address_map[address] = inst
                    counter += 1

        # map prologue and epilogue to dummy instructions
        for address in address_list:
            if address not in address_map:
                address_map[address] = Instruction("UNMAPPED", True)

        test_case.address_map = address_map

    def get_return_instruction(self) -> Instruction:
        return Instruction("RET", False, "General", True)

    def get_unconditional_jump_instruction(self) -> Instruction:
        return Instruction("C.J", False, "General", True)

    def create_pte(self, test_case: TestCase):
        """
        Pick a random PTE bit (among the permitted ones) and set/reset it
        """
        if not self.pte_bit_choices:  # no choices, so PTE should stay intact
            return

        pte_bit = random.choice(self.pte_bit_choices)
        if pte_bit[1]:
            mask_clear = 0xFFFFFFFFFFFFFFFF ^ (1 << pte_bit[0])
            mask_set = 0x0
        else:
            mask_clear = 0xFFFFFFFFFFFFFFFF
            mask_set = 0x0 | (1 << pte_bit[0])
        test_case.faulty_pte = PageTableModifier(mask_set, mask_clear)

    def parse_line(
        self,
        line: str,
        line_num: int,
        instruction_map: Dict[str, List[InstructionSpec]],
    ) -> Instruction:
        line = line
        # get name and possible specs
        words = line.split()
        name = ""
        specs: List[InstructionSpec] = []
        word = words[0]
        key = name + word
        specs = instruction_map.get(key, [])
        name += word
        if not specs:
            raise AsmParserException(line_num, f"Unknown instruction {line}")
        # instrumentation?
        is_instrumentation = line.endswith("# INSTRUMENTATION")

        # remove comments
        if "#" in line:
            line = re.search(r"(.*)#.*", line).group(1).strip()  # type: ignore

        # extract operands
        operands_raw = line.split(",")
        if operands_raw in [[""], ["C.NOP"], ["NOP"]]:  # no operands
            operands_raw = []
        else:  # clean the operands
            operands_raw = [o.strip() for o in operands_raw]
            operands_raw[0] = operands_raw[0].split(" ")[1]

        # find a matching spec
        matching_specs = []
        for spec_candidate in specs:
            if len(spec_candidate.operands) != len(operands_raw):
                continue
            match = True
            for op_id, op_raw in enumerate(operands_raw):
                op_spec = spec_candidate.operands[op_id]
                # RISC-V: CONDs always implicit
                # RISC-V: no FLAGS, no AGEN
                if "." == op_raw[0]:  # match label
                    if op_spec.type != OT.LABEL:
                        match = False
                        break
                    continue
                elif "(" in op_raw:  # match address
                    if (
                        op_spec.type != OT.MEM
                    ):  # RISC-V: OT.MEM= "IMM(MEM)" = MEM+IMM.value or "(MEM)"
                        match = False
                        break
                    continue
                elif op_raw[0] == "x" or op_raw[0] == "X":  # match reg
                    if op_spec.type != OT.REG:
                        match = False
                        break
                    elif op_raw not in self.target_desc.registers[op_spec.width]:
                        match = False
                        break
                    continue
                elif op_raw[0] == "f" or op_raw[0] == "F":  # match fpreg
                    if op_spec.type != OT.REG:
                        match = False
                        break
                    elif op_raw not in self.target_desc.registers[op_spec.width]:
                        match = False
                        break
                    continue
                # match immediate value
                elif (
                    re.match(r"^-?[0-9]+$", op_raw)
                    or re.match(r"^-?0x[0-9abcdef]+$", op_raw)
                    or re.match(r"^-?0b[01]+$", op_raw)
                    or re.match(r"^-?[0-9]+\ *[+-]\ *[0-9]+$", op_raw)
                ):
                    if op_spec.type != OT.IMM:
                        match = False
                        break
                    continue
                else:
                    match = False
            if match:
                matching_specs.append(spec_candidate)
        parser_assert(
            len(matching_specs) != 0,
            line_num,
            f"Could not find a matching spec for {line}",
        )

        # pick an instruction spec
        # Priority:
        # - LABEL specs
        # - specs with MEM and IMM
        spec: InstructionSpec = None
        for match_spec in matching_specs:
            if match_spec.category == "FORMAT":
                continue
            for op in match_spec.operands:
                if op.type == OT.LABEL:
                    spec = match_spec
                    break
                elif op.type == OT.MEM:
                    for reop in match_spec.operands:
                        if reop.type == OT.IMM:
                            spec = match_spec
                            break
        if not spec:
            spec = matching_specs[0]

        # generate a corresponding Instruction
        inst = Instruction.from_spec(spec, is_instrumentation)
        op: Operand
        for op_id, op_raw in enumerate(operands_raw):
            op_spec = spec.operands[op_id]
            if op_spec.type == OT.REG:
                op = RegisterOperand(op_raw, op_spec.width, op_spec.src, op_spec.dest)
            elif op_spec.type == OT.MEM:
                address_match = re.search(r"\((.*)\)", op_raw)
                parser_assert(
                    address_match is not None, line_num, "Invalid memory address"
                )
                address_reg = address_match.group(1)  # type: ignore
                mem_op = address_reg
                imm_match = re.search("(.*?)\s*\(", op_raw)
                if imm_match != "(":
                    mem_op = imm_match.string + address_reg
                op = MemoryOperand(mem_op, op_spec.width, op_spec.src, op_spec.dest)
            elif op_spec.type == OT.IMM:
                op = ImmediateOperand(op_raw, op_spec.width)
            elif op_spec.type == OT.LABEL:
                assert spec.control_flow
                op = LabelOperand(op_raw)
            else:
                raise AsmParserException(
                    line_num, f"Unknown spec operand type {op_spec.type}"
                )
            inst.operands.append(op)

        for op_spec in spec.implicit_operands:
            op = self.generate_operand(op_spec, inst)
            inst.implicit_operands.append(op)

        return inst


GPRs = ["x9", "x10", "x11", "x12", "x13", "x14"]


class RISCVZeroAndMemPass(Pass):
    def __init__(self, target_desc: RISCVTargetDesc):
        super().__init__()
        self.target_desc = target_desc

    def run_on_test_case(self, test_case: TestCase) -> None:
        for func in test_case.functions:
            for bb in func:
                if bb == func.entry:
                    continue

                # collect all instructions that require operand instrumentation
                nonz_instructions = []
                has_mem_operands = []
                for inst in bb:
                    if inst.get_reg_operands() or inst.get_mem_operands():
                        for operand in inst.get_reg_operands():
                            if operand.value == "NONZERO GPRS":
                                if inst not in nonz_instructions:
                                    nonz_instructions.append(inst)
                        for operand in inst.get_mem_operands():
                            if inst not in has_mem_operands:
                                has_mem_operands.append(inst)
                            if operand.value == "NONZERO GPRS":
                                operand.value = random.choice(GPRs)

                # if there are MEM operands, instantiate them properly
                # NOTE: instructions have 1 MEM operand at most
                for inst in has_mem_operands:
                    self.instantiate_mem(inst)
                # make nonzero-registers into proper registers which are never 0
                for inst in nonz_instructions:
                    self.unzero_reg(inst, bb)

    def instantiate_mem(self, instr: Instruction):
        """Instantiate MEM operands properly"""
        add_op: ImmediateOperand = None
        if instr.get_imm_operands() is not None:
            for imm_op in instr.get_explicit_imm_operands():
                add_op = imm_op
                instr.operands.remove(imm_op)
                break
            if instr.name in ["C.FLD", "C.FSD"]:  # special cases
                for imm_op in instr.get_imm_operands():
                    add_op = imm_op
                    instr.implicit_operands.remove(imm_op)
                    break
        if add_op is not None:
            # instr.get_mem_operands()[0].value = add_op.value+"(x30)"
            add_op.value = (
                str(int(add_op.value) * 8)
                if instr.name in ["C.FLD", "C.FSD", "C.FLDSP", "C.FSDSP"]
                else add_op.value
            )
            instr.get_mem_operands()[0].value = (
                add_op.value + "(" + instr.get_mem_operands()[0].value + ")"
            )
        else:
            # instr.get_mem_operands()[0].value = "(x30)"
            instr.get_mem_operands()[0].value = (
                "(" + instr.get_mem_operands()[0].value + ")"
            )
        return

    def unzero_reg(self, instr: Instruction, parent: BasicBlock):
        """Force nonzero gprs to not be zero"""
        for operand in instr.get_reg_operands():
            if operand.value == "NONZERO GPRS":
                operand.value = random.choice(GPRs)
                # Setting register to 1 (Shouldn't just add 1 to it, since there's a chance it holds a -1)
                zero_instr = (
                    Instruction("AND", True)
                    .add_op(
                        RegisterOperand(operand.value, operand.get_width(), True, True)
                    )
                    .add_op(RegisterOperand("x0", operand.get_width(), True, True))
                    .add_op(RegisterOperand("x0", operand.get_width(), True, True))
                )
                parent.insert_before(instr, zero_instr)
                add_instr = (
                    Instruction("ADDI", True)
                    .add_op(
                        RegisterOperand(operand.value, operand.get_width(), True, True)
                    )
                    .add_op(
                        RegisterOperand(operand.value, operand.get_width(), True, True)
                    )
                    .add_op(ImmediateOperand("1", 1))
                )
                parent.insert_before(instr, add_instr)

            else:
                continue

        # NOTE: MEM non-zero operands don't need to be set to 1, since they will be sandboxed anyway
        return


class RISCVSandboxPass(Pass):
    def __init__(self, target_desc: RISCVTargetDesc):
        super().__init__()
        input_memory_size = CONF.input_main_region_size + CONF.input_faulty_region_size
        mask_size = int(math.log(input_memory_size, 2))
        mask = int(
            ("0b" + "1" * mask_size + "0" * CONF.memory_access_zeroed_bits), 2
        )  # mask = int(mask_in_bits, 2)
        self.mask_upper_width = mask_size - 12 if mask_size > 12 else 0
        self.mask_lower_width = mask_size if mask_size < 12 else 12
        self.mask_lower = mask % (2**self.mask_lower_width) if mask_size >= 12 else mask
        if self.mask_lower > 2047:
            self.mask_lower = str(self.mask_lower - 4096)
            mask_upper_value = math.floor(mask / (2**self.mask_lower_width)) + 1
            self.mask_upper = str(mask_upper_value) if mask_size >= 12 else "0"
            if math.log(mask_upper_value, 2) == self.mask_upper_width:
                self.mask_upper_width += 1  # we want to adjust this value so that the corresponding IMM operand gets inserted properly later on
        else:
            self.mask_upper = (
                str(math.floor(mask / (2**self.mask_lower_width)))
                if mask_size >= 12
                else "0"
            )

        self.target_desc = target_desc
        self.atomic_set = [
            "SC.W",
            "LR.W",
            "AMOSWAP.W",
            "AMOADD.W",
            "AMOXOR.W",
            "AMOAND.W",
            "AMOOR.W",
            "AMOMIN.W",
            "AMOMAX.W",
            "AMOMINU.W",
            "AMOMAXU.W",
        ]
        self.mask_loaded = True  # we make sure we only load the mask into x31 once per test, in order to save time

    def run_on_test_case(self, test_case: TestCase) -> None:
        first_instruction = None
        first_block = None
        for func in test_case.functions:
            for bb in func:
                if bb == func.entry:
                    continue

                # collect all instructions that require sandboxing
                memory_instructions = []
                for inst in bb:
                    if first_instruction is None:
                        first_instruction = inst
                        first_block = bb
                    if inst.has_mem_operand(True):
                        memory_instructions.append(inst)

                # sandbox them
                if not self.mask_loaded:
                    self.load_mask(first_instruction, first_block)
                for inst in memory_instructions:
                    self.sandbox_memory_access(inst, bb, self.atomic_set)

    def load_mask(self, instr: Instruction, parent: BasicBlock):
        """Load the mask into x31"""
        apply_mask_0 = (
            Instruction("LUI", True)
            .add_op(RegisterOperand("x31", 64, True, True))
            .add_op(ImmediateOperand(self.mask_upper, self.mask_upper_width))
        )
        apply_mask_1 = (
            Instruction("ADDI", True)
            .add_op(RegisterOperand("x31", 64, True, True))
            .add_op(RegisterOperand("x31", 64, True, True))
            .add_op(ImmediateOperand(self.mask_lower, self.mask_lower_width))
        )
        parent.insert_before(instr, apply_mask_0)
        parent.insert_before(instr, apply_mask_1)
        self.mask_loaded = True

    def sandbox_memory_access(
        self, instr: Instruction, parent: BasicBlock, atomic_instructions: List
    ):
        """Force the memory accesses into the page starting from x30"""

        # This method:
        # 1. Shifts the address register to the left by the number of bits in the immediate of the MEM (when there is such an immediate) (SLLI)
        # 1* or if the instruction is atomic, the register in the MEM operand gets shifted 2 bits (alignment to 4 bytes) (SLLI)
        # 2. Applies the mask found in x31 to the register in the MEM operand (AND)
        # 2* and as stated, it forces memory accesses to start from x30 (ADD)
        # --> Depending on the situation, step 1 may or may not be applied.

        if instr.get_mem_operands()[0].value in ["x2", "(x2)"]:
            return  # the one case we don't want to sandbox regarding MEMs - x2 is the Stack Pointer of RISC-V, and it may appear as an implicit mem operand

        mem_operands = instr.get_mem_operands()
        if mem_operands:
            assert (
                len(mem_operands) == 1
            ), f"Unexpected instruction format: {instr.name}"
            mem_operand: Operand = mem_operands[0]
            address_reg = re.search(r"\((.*?)\)", mem_operand.value).group(
                1
            )  # may be <imm>(REG) or (REG)
            imm_mem_operand = (
                int(mem_operand.value.split("(")[0])
                if mem_operand.value[0] != "("
                else None
            )
            imm_mem_operand = (
                imm_mem_operand if imm_mem_operand != 0 else 1
            )  # imm value can't be 0 (not accepted by SLLI + can't process imm_effective_width)
            if imm_mem_operand is not None:
                # There's an immediate operand? make room for it
                imm_effective_width = (
                    int(math.ceil(math.log(imm_mem_operand, 2)))
                    if imm_mem_operand >= 0
                    else int(math.ceil(math.log((imm_mem_operand * -1), 2))) + 1
                )
                shift_left = (
                    Instruction("SLLI", True)
                    .add_op(RegisterOperand(address_reg, mem_operand.width, True, True))
                    .add_op(RegisterOperand(address_reg, mem_operand.width, True, True))
                    .add_op(
                        ImmediateOperand(str(imm_effective_width), imm_effective_width)
                    )
                )
                parent.insert_before(instr, shift_left)
            elif instr.name in atomic_instructions:
                # The instruction is atomic? Align it's address register to 4 bytes (all atomic instructions are from a 32-bit extension)
                # - Align (32 bit)
                align = (
                    Instruction("SLLI", True)
                    .add_op(RegisterOperand(address_reg, mem_operand.width, True, True))
                    .add_op(RegisterOperand(address_reg, mem_operand.width, True, True))
                    .add_op(ImmediateOperand("2", 2))
                )
                parent.insert_before(instr, align)
            # - Apply mask
            apply_mask_end = (
                Instruction("AND", True)
                .add_op(RegisterOperand(address_reg, mem_operand.width, True, True))
                .add_op(RegisterOperand(address_reg, mem_operand.width, True, True))
                .add_op(RegisterOperand("x31", mem_operand.width, True, True))
            )
            parent.insert_before(instr, apply_mask_end)
            # - And finally, add x30
            set_mem_addr = (
                Instruction("ADD", True)
                .add_op(RegisterOperand(address_reg, mem_operand.width, True, True))
                .add_op(RegisterOperand(address_reg, mem_operand.width, True, True))
                .add_op(RegisterOperand("x30", mem_operand.width, True, True))
            )
            parent.insert_before(instr, set_mem_addr)
            return

        raise GeneratorException(
            "Attempt to sandbox an instruction without memory operands"
        )

    # NOTE: RISC-V does not raise division by zero or division overflow exceptions https://five-embeddev.com/riscv-isa-manual/latest/m.html (Section 6.2)

    @staticmethod
    def requires_sandbox(inst: InstructionSpec):
        if inst.has_mem_operand:
            return True

        # may need more cases later on
        return False


class RISCVPrinter(Printer):

    prologue_template = [
        "FENCE rw, rw # instrumentation\n",
        ".test_case_enter:\n",
    ]
    epilogue_template = [
        ".test_case_exit:\n",
        "FENCE rw, rw # instrumentation\n",
    ]

    def print(self, test_case: TestCase, outfile: str) -> None:
        with open(outfile, "w") as f:
            # print prologue
            for line in self.prologue_template:
                f.write(line)

            # print the test case
            for func in test_case.functions:
                f.write(f"{func.name}:\n")
                for bb in func:
                    self.print_basic_block(bb, f)

            # print epilogue
            for line in self.epilogue_template:
                f.write(line)

        for i in self.prologue_template:
            if i[0] != ".":
                test_case.num_prologue_instructions += 1

    def print_basic_block(self, bb: BasicBlock, file):
        file.write(f"{bb.name}:\n")
        for inst in bb:
            file.write(self.instruction_to_str(inst) + "\n")
        for inst in bb.terminators:
            file.write(self.instruction_to_str(inst) + "\n")

    def instruction_to_str(self, inst: Instruction):
        operands = ", ".join([self.operand_to_str(op) for op in inst.operands])
        comment = "# instrumentation" if inst.is_instrumentation else ""
        return f"{inst.name} {operands} {comment}"

    def operand_to_str(self, op: Operand) -> str:
        return op.value.lower()


class RISCVRandomGenerator(RISCVGenerator, RandomGenerator):

    def __init__(self, instruction_set: InstructionSet, seed: int):
        super().__init__(instruction_set, seed)
