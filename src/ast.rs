pub type Program = Vec<TopLevel>;

#[derive(Debug, Clone)]
pub enum TopLevel {
    Binding {
        name: String,
        params: Vec<String>,
        body: Expr,
    },
    TypeAnnotation {
        name: String,
        ty: TypeExpr,
    },
}

#[derive(Debug, Clone)]
pub enum Expr {
    Lit(Literal),
    Var(String),
    App(Box<Expr>, Box<Expr>),
    BinOp {
        op: BinOpKind,
        left: Box<Expr>,
        right: Box<Expr>,
    },
    UnaryMinus(Box<Expr>),
    Pipe(Box<Expr>, Box<Expr>),
    If {
        cond: Box<Expr>,
        then: Box<Expr>,
        else_: Box<Expr>,
    },
    Let {
        name: String,
        params: Vec<String>,
        value: Box<Expr>,
        body: Box<Expr>,
    },
    /// rows[i] は i 行目の要素リスト。`;` で行を区切る
    TensorLit(Vec<Vec<Expr>>),
    Lambda {
        param: String,
        body: Box<Expr>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum Literal {
    Int(i64),
    Float(f64),
    Bool(bool),
}

#[derive(Debug, Clone, PartialEq)]
pub enum BinOpKind {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    MatMul,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

#[derive(Debug, Clone)]
pub enum TypeExpr {
    Named(String),
    Arrow(Box<TypeExpr>, Box<TypeExpr>),
    Tensor(Vec<Option<usize>>),
    App(Box<TypeExpr>, Box<TypeExpr>),
}
