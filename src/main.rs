use num_derive::FromPrimitive;
use num_traits::FromPrimitive;

type Register = u32;

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

#[derive(Debug, Clone)]
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
        let rs2_mask: u32    = 0b0000_0001_1111_0000_0000_0000_0000_0000;
        let rs1_mask: u32    = 0b0000_0000_0000_1111_1000_0000_0000_0000;
        let funct3_mask: u32 = 0b0000_0000_0000_0000_0111_0000_0000_0000;
        let rd_mask: u32     = 0b0000_0000_0000_0000_0000_1111_1000_0000;
        let opcode_mask: u32 = 0b0000_0000_0000_0000_0000_0000_0111_1111;

        Ok(RType {
            // 7 bits
            opcode: BaseOpcode::from_u32(opcode_mask & value).ok_or(Self::Error::InvalidInstructionError)?,
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

#[derive(Debug, Clone)]
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
        let imm_mask: u32    = 0b1111_1111_1111_0000_0000_0000_0000_0000;
        let rs1_mask: u32    = 0b0000_0000_0000_1111_1000_0000_0000_0000;
        let funct3_mask: u32 = 0b0000_0000_0000_0000_0111_0000_0000_0000;
        let rd_mask: u32     = 0b0000_0000_0000_0000_0000_1111_1000_0000;
        let opcode_mask: u32 = 0b0000_0000_0000_0000_0000_0000_0111_1111;

        Ok(IType {
            // 7 bits
            opcode: BaseOpcode::from_u32(opcode_mask & value).ok_or(Self::Error::InvalidInstructionError)?,
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

#[derive(Debug, Clone)]
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
        let rs2_mask: u32    = 0b0000_0001_1111_0000_0000_0000_0000_0000;
        let rs1_mask: u32    = 0b0000_0000_0000_1111_1000_0000_0000_0000;
        let funct3_mask: u32 = 0b0000_0000_0000_0000_0111_0000_0000_0000;
        let imm_lo_mask: u32 = 0b0000_0000_0000_0000_0000_1111_1000_0000;
        let opcode_mask: u32 = 0b0000_0000_0000_0000_0000_0000_0111_1111;

        // higher part of immediate is only last 7 bits thus shifting 25 bits to left
        // extracts the part, but as other bits are zero-cleared, it can be simply done by
        // shifting 20 bits to left and add lower 5 bits
        let imm = ((value & imm_hi_mask) >> 20) | ((value & imm_lo_mask >> 7));

        Ok(SType {
            // 7 bits
            opcode: BaseOpcode::from_u32(opcode_mask & value).ok_or(Self::Error::InvalidInstructionError)?,
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

#[derive(Debug, Clone)]
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
        let rs2_mask: u32    = 0b0000_0001_1111_0000_0000_0000_0000_0000;
        let rs1_mask: u32    = 0b0000_0000_0000_1111_1000_0000_0000_0000;
        let funct3_mask: u32 = 0b0000_0000_0000_0000_0111_0000_0000_0000;
        let imm_lo_mask: u32 = 0b0000_0000_0000_0000_0000_1111_1000_0000;
        let opcode_mask: u32 = 0b0000_0000_0000_0000_0000_0000_0111_1111;

        // TODO: this should be rewritten I guess, I need to figure out what
        // imm[12|10:5] and imm[12|10:5] stands for
        let imm = ((value & imm_hi_mask) >> 20) | ((value & imm_lo_mask >> 7));

        Ok(BType {
            // 7 bits
            opcode: BaseOpcode::from_u32(opcode_mask & value).ok_or(Self::Error::InvalidInstructionError)?,
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

#[derive(Debug, Clone)]
struct UType {
    opcode: BaseOpcode,
    rd: u8,
    imm: u32,
}

impl TryFrom<u32> for UType {
    type Error = TypeDecodeError;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        let imm_mask: u32    = 0b1111_1111_1111_1111_1111_0000_0000_0000;
        let rd_mask: u32     = 0b0000_0000_0000_0000_0000_1111_1000_0000;
        let opcode_mask: u32 = 0b0000_0000_0000_0000_0000_0000_0111_1111;

        Ok(UType {
            opcode: BaseOpcode::from_u32(opcode_mask & value).ok_or(Self::Error::InvalidInstructionError)?,
            rd: ((rd_mask & value) >> 7) as u8,
            imm: imm_mask & value
        })
    }
}

#[derive(Debug, Clone)]
struct JType {
    opcode: BaseOpcode,
    rd: u8,
    imm: u32,
}

impl TryFrom<u32> for JType {
    type Error = TypeDecodeError;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        let imm_mask: u32    = 0b1111_1111_1111_1111_1111_0000_0000_0000;
        let rd_mask: u32     = 0b0000_0000_0000_0000_0000_1111_1000_0000;
        let opcode_mask: u32 = 0b0000_0000_0000_0000_0000_0000_0111_1111;

        Ok(JType {
            opcode: BaseOpcode::from_u32(opcode_mask & value).ok_or(Self::Error::InvalidInstructionError)?,
            rd: ((rd_mask & value) >> 7) as u8,
            // TODO: I need to figure out what imm[20|10:1|11|19:12] stands for
            imm: imm_mask & value
        })
    }
}

#[derive(Debug, Clone)]
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
#[derive(Debug, Clone, FromPrimitive)]
#[repr(u8)]
enum BaseOpcode {
    // [6:5] = 0b00
    Load    = 0b0000011,
    LoadFp  = 0b0000111,
    MiscMem = 0b0001111,
    OpImm   = 0b0010011,
    AuiPc   = 0b0010111,
    OpImm32 = 0b0011011,

    // [6:5] = 0b01
    Store   = 0b0100011,
    StoreFp = 0b0100111,
    Amo     = 0b0101111,
    Op      = 0b0110011,
    Lui     = 0b0110111,
    Op32    = 0b0111011,

    // [6:5] = 0b10
    MAdd    = 0b1000011,
    MSub    = 0b1000111,
    NMSub   = 0b1001011,
    NMAdd   = 0b1001111,
    OpFp    = 0b1010011,
    OpV     = 0b1010111,

    // [6:5] = 0b11
    Branch  = 0b1100011,
    Jalr    = 0b1100111,
    Jal     = 0b1101111,
    System  = 0b1110011,
    OpVE    = 0b1110111,
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
        let base_opcode = BaseOpcode::from_u8(opcode_mask & value as u8).ok_or(ParseInstructionError::UnknownOpcodeError)?;
        
        use BaseOpcode::*;
        Ok(match base_opcode {
            Lui   => Instruction::Lui(UType::try_from(value)?),
            AuiPc => Instruction::AuiPc(UType::try_from(value)?),
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
