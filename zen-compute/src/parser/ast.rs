//! Abstract Syntax Tree definitions for CUDA

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Root AST node
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct Ast {
    pub items: Vec<Item>,
}

/// Top-level items in CUDA code
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub enum Item {
    /// Kernel function definition
    Kernel(KernelDef),
    /// Device function
    DeviceFunction(FunctionDef),
    /// Host function
    HostFunction(FunctionDef),
    /// Global variable
    GlobalVar(GlobalVar),
    /// Type definition
    TypeDef(TypeDef),
    /// Include directive
    Include(String),
}

/// CUDA kernel definition
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct KernelDef {
    pub name: String,
    pub params: Vec<Parameter>,
    pub body: Block,
    pub attributes: Vec<KernelAttribute>,
}

/// Kernel attributes (launch bounds, etc.)
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub enum KernelAttribute {
    LaunchBounds { max_threads: u32, min_blocks: Option<u32> },
    MaxRegisters(u32),
}

/// Function definition
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct FunctionDef {
    pub name: String,
    pub return_type: Type,
    pub params: Vec<Parameter>,
    pub body: Block,
    pub qualifiers: Vec<FunctionQualifier>,
}

/// Function qualifiers
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub enum FunctionQualifier {
    Device,
    Host,
    Global,
    Inline,
    NoInline,
}

/// Function parameter
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct Parameter {
    pub name: String,
    pub ty: Type,
    pub qualifiers: Vec<ParamQualifier>,
}

/// Parameter qualifiers
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub enum ParamQualifier {
    Const,
    Restrict,
    Volatile,
}

/// CUDA types
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub enum Type {
    /// Primitive types
    Void,
    Bool,
    Int(IntType),
    Float(FloatType),
    /// Pointer type
    Pointer(Box<Type>),
    /// Array type
    Array(Box<Type>, Option<usize>),
    /// Vector types (float4, int2, etc.)
    Vector(VectorType),
    /// User-defined type
    Named(String),
    /// Texture type
    Texture(TextureType),
}

/// Integer types
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub enum IntType {
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
}

/// Floating-point types
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub enum FloatType {
    F16,
    F32,
    F64,
}

/// Vector types
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct VectorType {
    pub element: Box<Type>,
    pub size: u8, // 1, 2, 3, or 4
}

/// Texture types
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct TextureType {
    pub dim: TextureDim,
    pub element: Box<Type>,
}

/// Texture dimensions
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub enum TextureDim {
    Tex1D,
    Tex2D,
    Tex3D,
    TexCube,
}

/// Statement types
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub enum Statement {
    /// Variable declaration
    VarDecl {
        name: String,
        ty: Type,
        init: Option<Expression>,
        storage: StorageClass,
    },
    /// Expression statement
    Expr(Expression),
    /// Block statement
    Block(Block),
    /// If statement
    If {
        condition: Expression,
        then_branch: Box<Statement>,
        else_branch: Option<Box<Statement>>,
    },
    /// For loop
    For {
        init: Option<Box<Statement>>,
        condition: Option<Expression>,
        update: Option<Expression>,
        body: Box<Statement>,
    },
    /// While loop
    While {
        condition: Expression,
        body: Box<Statement>,
    },
    /// Return statement
    Return(Option<Expression>),
    /// Break statement
    Break,
    /// Continue statement
    Continue,
    /// Synchronization
    SyncThreads,
}

/// Storage classes
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub enum StorageClass {
    Auto,
    Register,
    Shared,
    Global,
    Constant,
    Local,
}

/// Block of statements
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct Block {
    pub statements: Vec<Statement>,
}

/// Expression types
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub enum Expression {
    /// Literal values
    Literal(Literal),
    /// Variable reference
    Var(String),
    /// Binary operation
    Binary {
        op: BinaryOp,
        left: Box<Expression>,
        right: Box<Expression>,
    },
    /// Unary operation
    Unary {
        op: UnaryOp,
        expr: Box<Expression>,
    },
    /// Function call
    Call {
        name: String,
        args: Vec<Expression>,
    },
    /// Array access
    Index {
        array: Box<Expression>,
        index: Box<Expression>,
    },
    /// Member access
    Member {
        object: Box<Expression>,
        field: String,
    },
    /// Cast expression
    Cast {
        ty: Type,
        expr: Box<Expression>,
    },
    /// Thread index access
    ThreadIdx(Dimension),
    /// Block index access
    BlockIdx(Dimension),
    /// Block dimension access
    BlockDim(Dimension),
    /// Grid dimension access
    GridDim(Dimension),
    /// Warp-level primitives
    WarpPrimitive {
        op: WarpOp,
        args: Vec<Expression>,
    },
}

/// Dimensions for thread/block indexing
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum Dimension {
    X,
    Y,
    Z,
}

/// Literal values
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub enum Literal {
    Bool(bool),
    Int(i64),
    UInt(u64),
    Float(f64),
    String(String),
}

/// Binary operators
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    And,
    Or,
    Xor,
    Shl,
    Shr,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    LogicalAnd,
    LogicalOr,
    Assign,
}

/// Unary operators
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub enum UnaryOp {
    Not,
    Neg,
    BitNot,
    PreInc,
    PreDec,
    PostInc,
    PostDec,
    Deref,
    AddrOf,
}

/// Warp-level operations
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub enum WarpOp {
    Shuffle,
    ShuffleXor,
    ShuffleUp,
    ShuffleDown,
    Vote,
    Ballot,
    ActiveMask,
}

/// Global variable definition
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct GlobalVar {
    pub name: String,
    pub ty: Type,
    pub storage: StorageClass,
    pub init: Option<Expression>,
}

/// Type definition
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct TypeDef {
    pub name: String,
    pub ty: Type,
}