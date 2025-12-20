use num_derive::FromPrimitive;
use num_traits::FromPrimitive;

type Register = u32;

#[derive(Debug, Clone, Copy)]
struct OrigBitWidth(u32);

fn sign_extend(src: u64, bit_width: OrigBitWidth) -> u64 {
    let msb_mask = 2u64.pow(bit_width.0 - 1);
    if src & msb_mask > 1 {
        (std::u64::MAX - ((msb_mask << 1) - 1)) | src
    } else {
        !(std::u64::MAX - ((msb_mask << 1) - 1)) & src
    }
}

#[derive(Clone)]
struct Cpu {
    registers: [Register; 32],
    pc: Register,
}

impl std::fmt::Debug for Cpu {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut base = f.debug_struct("Cpu");
        self.registers
            .iter()
            .enumerate()
            .for_each(|(reg_num, register)| {
                base.field(&format!("x{reg_num}"), register);
            });

        base.field("pc", &self.pc).finish()
    }
}

#[derive(Debug, Clone)]
struct Tape {
    inner: Vec<Instruction>,
}

#[derive(Debug, Clone)]
struct Memory<const N: usize> {
    inner: [u8; N],
}

#[derive(Debug, Clone)]
struct Machine<const MEMORY_SIZE: usize> {
    cpu: Cpu,
    tape: Tape,
    memory: Memory<MEMORY_SIZE>,
}

impl<const N: usize> Machine<N> {
    pub fn new() -> Self {
        Machine {
            cpu: Cpu {
                registers: [0; 32],
                pc: 0,
            },
            tape: Tape { inner: Vec::new() },
            memory: Memory { inner: [0; N] },
        }
    }

    pub fn dump_cpu(&self) {
        dbg!(&self.cpu);
    }
}

#[derive(Debug, thiserror::Error)]
enum TypeDecodeError {
    #[error("not this instruction")]
    InvalidInstructionError,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct RType {
    opcode: BaseOpcode,
    rd: u8,
    funct3: u8,
    rs1: u8,
    rs2: u8,
    funct7: u8,
}

impl TryFrom<u32> for RType {
    type Error = TypeDecodeError;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        let funct7_mask: u32 = 0b1111_1110_0000_0000_0000_0000_0000_0000;
        let rs2_mask: u32 = 0b0000_0001_1111_0000_0000_0000_0000_0000;
        let rs1_mask: u32 = 0b0000_0000_0000_1111_1000_0000_0000_0000;
        let funct3_mask: u32 = 0b0000_0000_0000_0000_0111_0000_0000_0000;
        let rd_mask: u32 = 0b0000_0000_0000_0000_0000_1111_1000_0000;
        let opcode_mask: u32 = 0b0000_0000_0000_0000_0000_0000_0111_1111;

        Ok(RType {
            // 7 bits
            opcode: BaseOpcode::from_u32(opcode_mask & value)
                .ok_or(Self::Error::InvalidInstructionError)?,
            // 5 bits
            rd: ((rd_mask & value) >> 7) as u8,
            // 3 bits
            funct3: ((funct3_mask & value) >> 12) as u8,
            // 5 bits
            rs1: ((rs1_mask & value) >> 15) as u8,
            // 5 bits
            rs2: ((rs2_mask & value) >> 20) as u8,
            // 7 bits
            funct7: ((funct7_mask & value) >> 25) as u8,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct IType {
    opcode: BaseOpcode,
    rd: u8,
    funct3: u8,
    rs1: u8,
    imm: u16,
}

impl TryFrom<u32> for IType {
    type Error = TypeDecodeError;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        let imm_mask: u32 = 0b1111_1111_1111_0000_0000_0000_0000_0000;
        let rs1_mask: u32 = 0b0000_0000_0000_1111_1000_0000_0000_0000;
        let funct3_mask: u32 = 0b0000_0000_0000_0000_0111_0000_0000_0000;
        let rd_mask: u32 = 0b0000_0000_0000_0000_0000_1111_1000_0000;
        let opcode_mask: u32 = 0b0000_0000_0000_0000_0000_0000_0111_1111;

        Ok(IType {
            // 7 bits
            opcode: BaseOpcode::from_u32(opcode_mask & value)
                .ok_or(Self::Error::InvalidInstructionError)?,
            // 5 bits
            rd: ((rd_mask & value) >> 7) as u8,
            // 3 bits
            funct3: ((funct3_mask & value) >> 12) as u8,
            // 5 bits
            rs1: ((rs1_mask & value) >> 15) as u8,
            // 12 bits
            imm: ((imm_mask & value) >> 20) as u16,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SType {
    opcode: BaseOpcode,
    imm: u16,
    funct3: u8,
    rs1: u8,
    rs2: u8,
}

impl TryFrom<u32> for SType {
    type Error = TypeDecodeError;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        let imm_hi_mask: u32 = 0b1111_1110_0000_0000_0000_0000_0000_0000;
        let rs2_mask: u32 = 0b0000_0001_1111_0000_0000_0000_0000_0000;
        let rs1_mask: u32 = 0b0000_0000_0000_1111_1000_0000_0000_0000;
        let funct3_mask: u32 = 0b0000_0000_0000_0000_0111_0000_0000_0000;
        let imm_lo_mask: u32 = 0b0000_0000_0000_0000_0000_1111_1000_0000;
        let opcode_mask: u32 = 0b0000_0000_0000_0000_0000_0000_0111_1111;

        // higher part of immediate is only last 7 bits thus shifting 25 bits to left
        // extracts the part, but as other bits are zero-cleared, it can be simply done by
        // shifting 20 bits to left and add lower 5 bits
        let imm = ((value & imm_hi_mask) >> 20) | (value & imm_lo_mask >> 7);

        Ok(SType {
            // 7 bits
            opcode: BaseOpcode::from_u32(opcode_mask & value)
                .ok_or(Self::Error::InvalidInstructionError)?,
            // 5 bits (imm_lo)
            // 3 bits
            funct3: ((funct3_mask & value) >> 12) as u8,
            // 5 bits
            rs1: ((rs1_mask & value) >> 15) as u8,
            // 5 bits
            rs2: ((rs2_mask & value) >> 20) as u8,
            // 7 bits (plus 5 bits)
            imm: imm as u16,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct BType {
    opcode: BaseOpcode,
    imm: u16,
    funct3: u8,
    rs1: u8,
    rs2: u8,
}

impl TryFrom<u32> for BType {
    type Error = TypeDecodeError;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        let imm_hi_mask: u32 = 0b1111_1110_0000_0000_0000_0000_0000_0000;
        let rs2_mask: u32 = 0b0000_0001_1111_0000_0000_0000_0000_0000;
        let rs1_mask: u32 = 0b0000_0000_0000_1111_1000_0000_0000_0000;
        let funct3_mask: u32 = 0b0000_0000_0000_0000_0111_0000_0000_0000;
        let imm_lo_mask: u32 = 0b0000_0000_0000_0000_0000_1111_1000_0000;
        let opcode_mask: u32 = 0b0000_0000_0000_0000_0000_0000_0111_1111;

        // TODO: this should be rewritten I guess, I need to figure out what
        // imm[12|10:5] and imm[4:1|11] stands for
        let imm = ((value & imm_hi_mask) >> 20) | ((value & imm_lo_mask) >> 7);

        Ok(BType {
            // 7 bits
            opcode: BaseOpcode::from_u32(opcode_mask & value)
                .ok_or(Self::Error::InvalidInstructionError)?,
            // 5 bits (imm_lo)
            // 3 bits
            funct3: ((funct3_mask & value) >> 12) as u8,
            // 5 bits
            rs1: ((rs1_mask & value) >> 15) as u8,
            // 5 bits
            rs2: ((rs2_mask & value) >> 20) as u8,
            // 7 bits (plus 5 bits)
            imm: imm as u16,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct UType {
    opcode: BaseOpcode,
    rd: u8,
    imm: u32,
}

impl TryFrom<u32> for UType {
    type Error = TypeDecodeError;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        let imm_mask: u32 = 0b1111_1111_1111_1111_1111_0000_0000_0000;
        let rd_mask: u32 = 0b0000_0000_0000_0000_0000_1111_1000_0000;
        let opcode_mask: u32 = 0b0000_0000_0000_0000_0000_0000_0111_1111;

        Ok(UType {
            opcode: BaseOpcode::from_u32(opcode_mask & value)
                .ok_or(Self::Error::InvalidInstructionError)?,
            rd: ((rd_mask & value) >> 7) as u8,
            imm: imm_mask & value,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct JType {
    opcode: BaseOpcode,
    rd: u8,
    imm: u32,
}

impl TryFrom<u32> for JType {
    type Error = TypeDecodeError;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        let imm_mask: u32 = 0b1111_1111_1111_1111_1111_0000_0000_0000;
        let rd_mask: u32 = 0b0000_0000_0000_0000_0000_1111_1000_0000;
        let opcode_mask: u32 = 0b0000_0000_0000_0000_0000_0000_0111_1111;

        Ok(JType {
            opcode: BaseOpcode::from_u32(opcode_mask & value)
                .ok_or(Self::Error::InvalidInstructionError)?,
            rd: ((rd_mask & value) >> 7) as u8,
            // TODO: I need to figure out what imm[20|10:1|11|19:12] stands for
            imm: imm_mask & value,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[repr(u8)]
enum Instruction {
    Lui(UType),
    AuiPc(UType),
    Jal(JType),
    Jalr(IType),

    // Branches
    Beq(BType),
    Bne(BType),
    Blt(BType),
    Bge(BType),
    Bltu(BType),
    Bgeu(BType),

    // Loads
    Lb(IType),
    Lh(IType),
    Lw(IType),
    Lbu(IType),
    Lhu(IType),

    // Subs
    Sb(SType),
    Sh(SType),
    Sw(SType),

    // Arithmetics and Bits with immediate
    Addi(IType),
    Slti(IType),
    Sltiu(IType),
    Xori(IType),
    Ori(IType),
    Andi(IType),
    Slli(RType),
    Srli(RType),
    Srai(RType),

    // Arithmetics and Bits (Direct Registers)
    Add(RType),
    Sub(RType),
    Sll(RType),
    Slt(RType),
    Sltu(RType),
    Xor(RType),
    Srl(RType),
    Sra(RType),
    Or(RType),
    And(RType),

    Nop,
}

/// Base OpCode(ISA Manual Page 553, Table 79)
/// For instructions that have [1:0] bits equal to 0b11,
/// only when [4:2] = 0b111 they are NON-32bit instructions.
/// Here I assume all instructions are 32-bit length
/// as I'm writing an RV32I emulator.
#[derive(Debug, Clone, FromPrimitive, PartialEq, Eq)]
#[repr(u8)]
enum BaseOpcode {
    // [6:5] = 0b00
    Load = 0b0000011,
    LoadFp = 0b0000111,
    MiscMem = 0b0001111,
    OpImm = 0b0010011,
    AuiPc = 0b0010111,
    OpImm32 = 0b0011011,

    // [6:5] = 0b01
    Store = 0b0100011,
    StoreFp = 0b0100111,
    Amo = 0b0101111,
    Op = 0b0110011,
    Lui = 0b0110111,
    Op32 = 0b0111011,

    // [6:5] = 0b10
    MAdd = 0b1000011,
    MSub = 0b1000111,
    NMSub = 0b1001011,
    NMAdd = 0b1001111,
    OpFp = 0b1010011,
    OpV = 0b1010111,

    // [6:5] = 0b11
    Branch = 0b1100011,
    Jalr = 0b1100111,
    Jal = 0b1101111,
    System = 0b1110011,
    OpVE = 0b1110111,
}

fn parse_opcodes(tape: &[u32]) -> Vec<String> {
    tape.iter()
        .map(|instruction| {
            let mask: u32 = 0b1111111;
            let opcode = mask & instruction;
            let argument = instruction >> 7;
            format!("opcode: 0b{:0>7b}, argument: 0b{:0>25b}", opcode, argument)
        })
        .collect::<Vec<String>>()
}

#[derive(Debug, Clone, thiserror::Error)]
enum ParseInstructionError {
    #[error("somehow failed")]
    GenericError,
    #[error("reserved/unknown opcode")]
    UnknownOpcodeError,
}

impl From<TypeDecodeError> for ParseInstructionError {
    fn from(_value: TypeDecodeError) -> Self {
        ParseInstructionError::GenericError
    }
}

impl TryFrom<u32> for Instruction {
    type Error = ParseInstructionError;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        let opcode_mask = 0b1111111u8;
        let base_opcode = BaseOpcode::from_u8(opcode_mask & value as u8)
            .ok_or(ParseInstructionError::UnknownOpcodeError)?;

        use BaseOpcode::*;
        Ok(match base_opcode {
            Lui => Instruction::Lui(UType::try_from(value)?),
            AuiPc => Instruction::AuiPc(UType::try_from(value)?),
            Jal => Instruction::Jal(JType::try_from(value)?),
            Jalr => Instruction::Jalr(IType::try_from(value)?),
            Branch => {
                let inner = BType::try_from(value)?;
                match inner.funct3 {
                    0b000 => Instruction::Beq(inner),
                    0b001 => Instruction::Bne(inner),
                    0b100 => Instruction::Blt(inner),
                    0b101 => Instruction::Bge(inner),
                    0b110 => Instruction::Bltu(inner),
                    0b111 => Instruction::Bgeu(inner),
                    _ => Instruction::Nop,
                }
            }
            Load => {
                let inner = IType::try_from(value)?;
                match inner.funct3 {
                    0b000 => Instruction::Lb(inner),
                    0b001 => Instruction::Lh(inner),
                    0b010 => Instruction::Lw(inner),
                    0b100 => Instruction::Lbu(inner),
                    0b101 => Instruction::Lhu(inner),
                    _ => Instruction::Nop,
                }
            }
            Store => {
                let inner: SType = SType::try_from(value)?;
                match inner.funct3 {
                    0b000 => Instruction::Sb(inner),
                    0b001 => Instruction::Sh(inner),
                    0b010 => Instruction::Sw(inner),
                    _ => Instruction::Nop,
                }
            }
            OpImm => {
                let inner = IType::try_from(value)?;
                match inner.funct3 {
                    0b000 => Instruction::Addi(inner),
                    0b010 => Instruction::Slti(inner),
                    0b011 => Instruction::Sltiu(inner),
                    0b100 => Instruction::Xori(inner),
                    0b110 => Instruction::Ori(inner),
                    0b111 => Instruction::Andi(inner),

                    // Ah shit, here we go again....
                    0b001 | 0b101 => {
                        let inner = RType::try_from(value)?;
                        match (inner.funct3, inner.funct7) {
                            (0b001, 0b0000000) => Instruction::Slli(inner),
                            (0b101, 0b0000000) => Instruction::Srli(inner),
                            (0b101, 0b0100000) => Instruction::Srai(inner),
                            _ => Instruction::Nop,
                        }
                    }

                    _ => unreachable!(), // as we have 3-bit value here
                }
            }
            Op => {
                let inner = RType::try_from(value)?;
                match (inner.funct3, inner.funct7) {
                    (0b000, 0b0000000) => Instruction::Add(inner),
                    (0b000, 0b0100000) => Instruction::Sub(inner),
                    (0b001, 0b0000000) => Instruction::Sll(inner),
                    (0b010, 0b0000000) => Instruction::Slt(inner),
                    (0b011, 0b0000000) => Instruction::Sltu(inner),
                    (0b100, 0b0000000) => Instruction::Xor(inner),
                    (0b101, 0b0000000) => Instruction::Srl(inner),
                    (0b101, 0b0100000) => Instruction::Sra(inner),
                    (0b110, 0b0000000) => Instruction::Or(inner),
                    (0b111, 0b0000000) => Instruction::And(inner),
                    _ => Instruction::Nop,
                }
            }
            _ => Instruction::Nop,
        })
    }
}

fn main() {
    let machine = Machine::<{ 32usize << 10 }>::new();
    machine.dump_cpu();
    let tape = vec![0xfff40413, 0x00008067];
    let parsed_tape = parse_opcodes(&tape);
    dbg!(parsed_tape);
}

#[cfg(test)]
mod tests {
    use crate::{Instruction, OrigBitWidth, sign_extend};
    use anyhow::Context;
    use goblin::elf::Elf;

    const RVTESTS_DIR: &str = "riscv-tests/isa/";

    fn get_insts_from_elf(name: &str, begin_addr: usize, count: usize) -> anyhow::Result<Vec<u8>> {
        let data = std::fs::read(std::path::PathBuf::from(RVTESTS_DIR).join(name)).with_context(|| format!("failed to read from specified file: {name}"))?;
        let elf = Elf::parse(&data).with_context(|| "failed to parse elf")?;
        let text_init_section = elf
            .section_headers
            .iter()
            .find(|section_header| {
                elf.shdr_strtab.get_at(section_header.sh_name) == Some(".text.init")
            })
            .with_context(|| "unable to find `.text.init` section")?;

        let start_addr = text_init_section.sh_offset as usize + begin_addr;
        let end_addr = start_addr + 4 * count;

        Ok(data[start_addr..end_addr].to_owned())
    }

    #[test]
    fn test_sign_extend() {
        let src = 0b0001_0000;
        let bw = OrigBitWidth(5);
        assert_eq!(
            sign_extend(src, bw),
            0b11111111_11111111_11111111_11111111_11111111_11111111_11111111_11110000
        );

        let src = 0b0001_0000;
        let bw = OrigBitWidth(6);
        assert_eq!(
            sign_extend(src, bw),
            0b00000000_00000000_00000000_00000000_00000000_00000000_00000000_00010000
        );
    }

    #[test]
    fn test_parse_addi() {
        let test2_function_section = get_insts_from_elf("rv64ui-p-addi", 0x190, 5).expect("error getting section from elf");

        let expected_insts = [
            Instruction::Addi(crate::IType {
                rd: 3,
                rs1: 0,
                funct3: 0b000,
                imm: 2,
                opcode: crate::BaseOpcode::OpImm,
            }),
            Instruction::Addi(crate::IType {
                rd: 13,
                rs1: 0,
                funct3: 0b000,
                imm: 0,
                opcode: crate::BaseOpcode::OpImm,
            }),
            Instruction::Addi(crate::IType {
                rd: 14,
                rs1: 13,
                funct3: 0b000,
                imm: 0,
                opcode: crate::BaseOpcode::OpImm,
            }),
            Instruction::Addi(crate::IType {
                rd: 7,
                rs1: 0,
                funct3: 0b000,
                imm: 0,
                opcode: crate::BaseOpcode::OpImm,
            }),
            Instruction::Bne(crate::BType {
                opcode: crate::BaseOpcode::Branch,
                imm: 644,
                funct3: 0b001,
                rs1: 14,
                rs2: 7,
            }),
        ];

        test2_function_section
            .chunks(4)
            .map(|inst| {
                let num = u32::from_le_bytes(inst.try_into().unwrap());
                Instruction::try_from(num).unwrap()
            })
            .zip(expected_insts.iter())
            .for_each(|(actual, expected)| {
                assert_eq!(actual, *expected);
            });
    }

    #[test]
    fn test_parse_add() {
        let test2_function_section = get_insts_from_elf("rv64ui-p-add", 0x190, 6).expect("error getting section from elf");

        let expected_insts = [
            Instruction::Addi(crate::IType {
                rd: 3,
                rs1: 0,
                funct3: 0b000,
                imm: 2,
                opcode: crate::BaseOpcode::OpImm,
            }),
            Instruction::Addi(crate::IType {
                rd: 11,
                rs1: 0,
                funct3: 0b000,
                imm: 0,
                opcode: crate::BaseOpcode::OpImm,
            }),
            Instruction::Addi(crate::IType {
                rd: 12,
                rs1: 0,
                funct3: 0b000,
                imm: 0,
                opcode: crate::BaseOpcode::OpImm,
            }),
            Instruction::Add(crate::RType {
                funct7: 0b0000000,
                rd: 14,
                rs1: 11,
                funct3: 0b000,
                rs2: 12,
                opcode: crate::BaseOpcode::Op,
            }),
            Instruction::Addi(crate::IType {
                rd: 7,
                rs1: 0,
                funct3: 0b000,
                imm: 0,
                opcode: crate::BaseOpcode::OpImm,
            }),
            Instruction::Bne(crate::BType {
                opcode: crate::BaseOpcode::Branch,
                imm: 1248,
                funct3: 0b001,
                rs1: 14,
                rs2: 7,
            }),
        ];

        test2_function_section
            .chunks(4)
            .map(|inst| {
                let num = u32::from_le_bytes(inst.try_into().unwrap());
                Instruction::try_from(num).unwrap()
            })
            .zip(expected_insts.iter())
            .for_each(|(actual, expected)| {
                assert_eq!(actual, *expected);
            });
    }
}
