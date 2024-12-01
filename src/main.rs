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

#[derive(Debug, Clone)]
struct RType {
    opcode: BaseOpcode,
    rd: u8,
    funct3: u8,
    rs1: u8,
    rs2: u8,
    funct7: u8,
}

#[derive(Debug, Clone)]
struct IType {
    opcode: BaseOpcode,
    rd: u8,
    funct3: u8,
    rs1: u8,
    imm: u16,
}

#[derive(Debug, Clone)]
struct SType {
    opcode: BaseOpcode,
    imm: u16,
    funct3: u8,
    rs1: u8,
    rs2: u8,
}

#[derive(Debug, Clone)]
struct BType {
    opcode: BaseOpcode,
    imm: u16,
    funct3: u8,
    rs1: u8,
    rs2: u8,
}

#[derive(Debug, Clone)]
struct UType {
    opcode: BaseOpcode,
    rd: u8,
    imm: u32,
}

#[derive(Debug, Clone)]
struct JType {
    opcode: BaseOpcode,
    rd: u8,
    imm: u32,
}

#[derive(Debug, Clone)]
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

    // wakaran
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
/// Here I assume all Instructions are 32-bit ones
/// as I'm writing RV32I emulator.
#[derive(Debug, Clone, FromPrimitive)]
#[repr(u8)]
enum BaseOpcode {
    // [6:5] = 0b00
    Load    = 0b00000,
    LoadFp  = 0b00001,
    MiscMem = 0b00011,
    OpImm   = 0b00100,
    AuiPc   = 0b00101,
    OpImm32 = 0b00110,

    // [6:5] = 0b01
    Store   = 0b01000,
    StoreFp = 0b01001,
    Amo     = 0b01011,
    Op      = 0b01100,
    Lui     = 0b01101,
    Op32    = 0b01110,

    // [6:5] = 0b10
    MAdd    = 0b10000,
    MSub    = 0b10001,
    NMSub   = 0b10010,
    NMAdd   = 0b10011,
    OpFp    = 0b10100,
    OpV     = 0b10101,

    // [6:5] = 0b11
    Branch  = 0b11000,
    JALR    = 0b11001,
    JAL     = 0b11011,
    System  = 0b11100,
    OpVE    = 0b11101,
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

impl TryFrom<u32> for Instruction {
    type Error = ParseInstructionError;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        let opcode_mask = 0b1111111u8;
        let base_opcode = BaseOpcode::from_u8(opcode_mask & value as u8).ok_or(ParseInstructionError::UnknownOpcodeError)?;
        
        match base_opcode {
            
        }
    }
}

fn main() {
    let machine = Machine::<{ 32usize << 10 }>::new();
    machine.dump_cpu();
    let tape = vec![0xfff40413, 0x00008067];
    let parsed_tape = parse_opcodes(&tape);
    dbg!(parsed_tape);
}
