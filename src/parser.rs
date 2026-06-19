/// Haplo パーサ（再帰下降）
use crate::ast::*;
use crate::lexer::{Span, Token, TokenKind};

#[derive(Debug)]
pub enum ParseError {
    UnexpectedToken {
        got: String,
        expected: &'static str,
        span: Span,
    },
    UnexpectedEof {
        expected: &'static str,
    },
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParseError::UnexpectedToken {
                got,
                expected,
                span,
            } => write!(
                f,
                "予期しないトークン {:?} ({}:{}) — {} を期待",
                got, span.line, span.col, expected
            ),
            ParseError::UnexpectedEof { expected } => {
                write!(f, "予期しないファイル末尾 — {} を期待", expected)
            }
        }
    }
}

pub struct Parser<'a> {
    tokens: &'a [Token],
    pos: usize,
}

impl<'a> Parser<'a> {
    fn new(tokens: &'a [Token]) -> Self {
        Parser { tokens, pos: 0 }
    }

    fn peek(&self) -> &TokenKind {
        &self.tokens[self.pos].kind
    }

    fn peek_token(&self) -> &Token {
        &self.tokens[self.pos]
    }

    fn advance(&mut self) -> &Token {
        let t = &self.tokens[self.pos];
        if self.pos + 1 < self.tokens.len() {
            self.pos += 1;
        }
        t
    }

    fn expect(&mut self, expected_kind: &TokenKind, label: &'static str) -> Result<&Token, ParseError> {
        if self.peek() == expected_kind {
            Ok(self.advance())
        } else {
            let tok = self.peek_token();
            Err(ParseError::UnexpectedToken {
                got: format!("{:?}", tok.kind),
                expected: label,
                span: tok.span.clone(),
            })
        }
    }

    fn eat_newlines(&mut self) {
        while matches!(self.peek(), TokenKind::Newline) {
            self.advance();
        }
    }

    fn at_eof(&self) -> bool {
        matches!(self.peek(), TokenKind::Eof)
    }

    // ----------------------------------------------------------------
    // トップレベル
    // ----------------------------------------------------------------

    fn parse_program(&mut self) -> Result<Program, ParseError> {
        let mut items = Vec::new();
        self.eat_newlines();
        while !self.at_eof() {
            items.push(self.parse_top_level()?);
            // 改行またはEOFで区切る
            while matches!(self.peek(), TokenKind::Newline) {
                self.advance();
            }
        }
        Ok(items)
    }

    fn parse_top_level(&mut self) -> Result<TopLevel, ParseError> {
        // `name : Type` または `name param* = expr`
        let name = match self.peek().clone() {
            TokenKind::Ident(n) => {
                self.advance();
                n
            }
            _ => {
                let tok = self.peek_token();
                return Err(ParseError::UnexpectedToken {
                    got: format!("{:?}", tok.kind),
                    expected: "識別子（トップレベル定義名）",
                    span: tok.span.clone(),
                });
            }
        };

        if matches!(self.peek(), TokenKind::Colon) {
            // 型注釈
            self.advance(); // consume `:`
            let ty = self.parse_type_expr()?;
            return Ok(TopLevel::TypeAnnotation { name, ty });
        }

        // パラメータを収集
        let mut params = Vec::new();
        while let TokenKind::Ident(p) = self.peek().clone() {
            params.push(p);
            self.advance();
        }

        self.expect(&TokenKind::Eq, "=")?;
        self.eat_newlines();
        let body = self.parse_expr()?;

        Ok(TopLevel::Binding { name, params, body })
    }

    // ----------------------------------------------------------------
    // 型式
    // ----------------------------------------------------------------

    fn parse_type_expr(&mut self) -> Result<TypeExpr, ParseError> {
        let mut t = self.parse_type_atom()?;
        while matches!(self.peek(), TokenKind::Arrow) {
            self.advance();
            let rhs = self.parse_type_atom()?;
            t = TypeExpr::Arrow(Box::new(t), Box::new(rhs));
        }
        Ok(t)
    }

    fn parse_type_atom(&mut self) -> Result<TypeExpr, ParseError> {
        match self.peek().clone() {
            TokenKind::Ident(name) => {
                self.advance();
                if name == "Tensor" && matches!(self.peek(), TokenKind::LBrack) {
                    self.advance(); // consume `[`
                    let mut dims = Vec::new();
                    loop {
                        match self.peek().clone() {
                            TokenKind::Int(n) => {
                                dims.push(Some(n as usize));
                                self.advance();
                            }
                            TokenKind::Ident(_) => {
                                dims.push(None); // 次元変数（P0 では無視）
                                self.advance();
                            }
                            _ => {}
                        }
                        match self.peek() {
                            TokenKind::Comma => {
                                self.advance();
                            }
                            TokenKind::RBrack => {
                                self.advance();
                                break;
                            }
                            _ => {
                                let tok = self.peek_token();
                                return Err(ParseError::UnexpectedToken {
                                    got: format!("{:?}", tok.kind),
                                    expected: "`,` または `]`",
                                    span: tok.span.clone(),
                                });
                            }
                        }
                    }
                    Ok(TypeExpr::Tensor(dims))
                } else {
                    // 型適用の引数（次元変数など）をスキャン
                    let mut ty = TypeExpr::Named(name);
                    // 型引数がある場合（例: `Vec n`）
                    while let TokenKind::Ident(arg) = self.peek().clone() {
                        self.advance();
                        ty = TypeExpr::App(Box::new(ty), Box::new(TypeExpr::Named(arg)));
                    }
                    Ok(ty)
                }
            }
            TokenKind::LParen => {
                self.advance();
                let t = self.parse_type_expr()?;
                self.expect(&TokenKind::RParen, ")")?;
                Ok(t)
            }
            _ => {
                let tok = self.peek_token();
                Err(ParseError::UnexpectedToken {
                    got: format!("{:?}", tok.kind),
                    expected: "型式",
                    span: tok.span.clone(),
                })
            }
        }
    }

    // ----------------------------------------------------------------
    // 式 — 優先順位の低い順
    // ----------------------------------------------------------------

    fn parse_expr(&mut self) -> Result<Expr, ParseError> {
        self.parse_pipe()
    }

    /// `|>` — 左結合
    fn parse_pipe(&mut self) -> Result<Expr, ParseError> {
        let mut expr = self.parse_comparison()?;
        while matches!(self.peek(), TokenKind::Pipe) {
            self.advance();
            self.eat_newlines();
            let rhs = self.parse_comparison()?;
            expr = Expr::Pipe(Box::new(expr), Box::new(rhs));
        }
        Ok(expr)
    }

    /// 比較演算子 — 非結合
    fn parse_comparison(&mut self) -> Result<Expr, ParseError> {
        let lhs = self.parse_additive()?;
        let op = match self.peek() {
            TokenKind::EqEq => BinOpKind::Eq,
            TokenKind::BangEq => BinOpKind::Ne,
            TokenKind::Lt => BinOpKind::Lt,
            TokenKind::Le => BinOpKind::Le,
            TokenKind::Gt => BinOpKind::Gt,
            TokenKind::Ge => BinOpKind::Ge,
            _ => return Ok(lhs),
        };
        self.advance();
        self.eat_newlines();
        let rhs = self.parse_additive()?;
        Ok(Expr::BinOp {
            op,
            left: Box::new(lhs),
            right: Box::new(rhs),
        })
    }

    /// `+` `-` — 左結合
    fn parse_additive(&mut self) -> Result<Expr, ParseError> {
        let mut expr = self.parse_matmul()?;
        loop {
            let op = match self.peek() {
                TokenKind::Plus => BinOpKind::Add,
                TokenKind::Minus => BinOpKind::Sub,
                _ => break,
            };
            self.advance();
            self.eat_newlines();
            let rhs = self.parse_matmul()?;
            expr = Expr::BinOp {
                op,
                left: Box::new(expr),
                right: Box::new(rhs),
            };
        }
        Ok(expr)
    }

    /// `@` — 左結合
    fn parse_matmul(&mut self) -> Result<Expr, ParseError> {
        let mut expr = self.parse_multiplicative()?;
        while matches!(self.peek(), TokenKind::At) {
            self.advance();
            self.eat_newlines();
            let rhs = self.parse_multiplicative()?;
            expr = Expr::BinOp {
                op: BinOpKind::MatMul,
                left: Box::new(expr),
                right: Box::new(rhs),
            };
        }
        Ok(expr)
    }

    /// `*` `/` — 左結合
    fn parse_multiplicative(&mut self) -> Result<Expr, ParseError> {
        let mut expr = self.parse_power()?;
        loop {
            let op = match self.peek() {
                TokenKind::Star => BinOpKind::Mul,
                TokenKind::Slash => BinOpKind::Div,
                _ => break,
            };
            self.advance();
            self.eat_newlines();
            let rhs = self.parse_power()?;
            expr = Expr::BinOp {
                op,
                left: Box::new(expr),
                right: Box::new(rhs),
            };
        }
        Ok(expr)
    }

    /// `^` — 右結合
    fn parse_power(&mut self) -> Result<Expr, ParseError> {
        let base = self.parse_unary()?;
        if matches!(self.peek(), TokenKind::Caret) {
            self.advance();
            self.eat_newlines();
            let exp = self.parse_power()?; // 右結合
            Ok(Expr::BinOp {
                op: BinOpKind::Pow,
                left: Box::new(base),
                right: Box::new(exp),
            })
        } else {
            Ok(base)
        }
    }

    /// 単項 `-`
    fn parse_unary(&mut self) -> Result<Expr, ParseError> {
        if matches!(self.peek(), TokenKind::Minus) {
            self.advance();
            let e = self.parse_application()?;
            Ok(Expr::UnaryMinus(Box::new(e)))
        } else {
            self.parse_application()
        }
    }

    /// 関数適用（空白区切り、左結合）
    fn parse_application(&mut self) -> Result<Expr, ParseError> {
        let mut expr = self.parse_atom()?;
        loop {
            if self.next_starts_atom() {
                let arg = self.parse_atom()?;
                expr = Expr::App(Box::new(expr), Box::new(arg));
            } else {
                break;
            }
        }
        Ok(expr)
    }

    fn next_starts_atom(&self) -> bool {
        // Minus は含めない: `10 - 3` を App(10, -3) ではなく BinOp(Sub, 10, 3) にする。
        // 負数を引数に渡す場合は括弧を使う: `f (-3)`
        matches!(
            self.peek(),
            TokenKind::Int(_)
                | TokenKind::Float(_)
                | TokenKind::Bool(_)
                | TokenKind::Ident(_)
                | TokenKind::LParen
                | TokenKind::LBrack
                | TokenKind::Let
                | TokenKind::If
        )
    }

    /// アトム
    fn parse_atom(&mut self) -> Result<Expr, ParseError> {
        match self.peek().clone() {
            TokenKind::Int(n) => {
                self.advance();
                Ok(Expr::Lit(Literal::Int(n)))
            }
            TokenKind::Float(x) => {
                self.advance();
                Ok(Expr::Lit(Literal::Float(x)))
            }
            TokenKind::Bool(b) => {
                self.advance();
                Ok(Expr::Lit(Literal::Bool(b)))
            }
            TokenKind::Ident(name) => {
                self.advance();
                Ok(Expr::Var(name))
            }
            TokenKind::Let => self.parse_let_expr(),
            TokenKind::If => self.parse_if_expr(),
            TokenKind::LParen => {
                self.advance(); // consume `(`
                self.eat_newlines();
                let e = self.parse_expr()?;
                self.eat_newlines();
                self.expect(&TokenKind::RParen, ")")?;
                Ok(e)
            }
            TokenKind::LBrack => self.parse_tensor_lit(),
            TokenKind::Minus => {
                // atom 内の単項マイナス（parse_application から再帰で来た場合）
                self.advance();
                let e = self.parse_atom()?;
                Ok(Expr::UnaryMinus(Box::new(e)))
            }
            _ => {
                let tok = self.peek_token();
                Err(ParseError::UnexpectedToken {
                    got: format!("{:?}", tok.kind),
                    expected: "式",
                    span: tok.span.clone(),
                })
            }
        }
    }

    fn parse_let_expr(&mut self) -> Result<Expr, ParseError> {
        self.expect(&TokenKind::Let, "let")?;
        let name = match self.peek().clone() {
            TokenKind::Ident(n) => {
                self.advance();
                n
            }
            _ => {
                let tok = self.peek_token();
                return Err(ParseError::UnexpectedToken {
                    got: format!("{:?}", tok.kind),
                    expected: "識別子",
                    span: tok.span.clone(),
                });
            }
        };

        // let 内の関数パラメータ
        let mut params = Vec::new();
        while let TokenKind::Ident(p) = self.peek().clone() {
            params.push(p);
            self.advance();
        }

        self.expect(&TokenKind::Eq, "=")?;
        self.eat_newlines();
        let value = self.parse_expr()?;
        self.eat_newlines();
        self.expect(&TokenKind::In, "in")?;
        self.eat_newlines();
        let body = self.parse_expr()?;

        Ok(Expr::Let {
            name,
            params,
            value: Box::new(value),
            body: Box::new(body),
        })
    }

    fn parse_if_expr(&mut self) -> Result<Expr, ParseError> {
        self.expect(&TokenKind::If, "if")?;
        self.eat_newlines();
        let cond = self.parse_expr()?;
        self.eat_newlines();
        self.expect(&TokenKind::Then, "then")?;
        self.eat_newlines();
        let then = self.parse_expr()?;
        self.eat_newlines();
        self.expect(&TokenKind::Else, "else")?;
        self.eat_newlines();
        let else_ = self.parse_expr()?;
        Ok(Expr::If {
            cond: Box::new(cond),
            then: Box::new(then),
            else_: Box::new(else_),
        })
    }

    fn parse_tensor_lit(&mut self) -> Result<Expr, ParseError> {
        self.expect(&TokenKind::LBrack, "[")?;
        self.eat_newlines();

        if matches!(self.peek(), TokenKind::RBrack) {
            self.advance();
            return Ok(Expr::TensorLit(vec![]));
        }

        let mut rows: Vec<Vec<Expr>> = vec![vec![]];

        loop {
            self.eat_newlines();
            match self.peek() {
                TokenKind::RBrack => {
                    self.advance();
                    break;
                }
                TokenKind::Comma => {
                    self.advance();
                    self.eat_newlines();
                }
                TokenKind::Semicolon => {
                    self.advance();
                    self.eat_newlines();
                    rows.push(vec![]);
                }
                _ => {
                    let e = self.parse_expr()?;
                    rows.last_mut().unwrap().push(e);
                }
            }
        }

        Ok(Expr::TensorLit(rows))
    }
}

pub fn parse(tokens: &[Token]) -> Result<Program, ParseError> {
    let mut parser = Parser::new(tokens);
    parser.parse_program()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::lex;

    fn parse_str(src: &str) -> Program {
        let tokens = lex(src).expect("lex error");
        parse(&tokens).expect("parse error")
    }

    fn parse_expr_str(src: &str) -> Expr {
        let src = format!("main = {}", src);
        let prog = parse_str(&src);
        match prog.into_iter().next().unwrap() {
            TopLevel::Binding { body, .. } => body,
            _ => panic!("expected binding"),
        }
    }

    #[test]
    fn test_parse_simple_binding() {
        let prog = parse_str("main = 42");
        assert_eq!(prog.len(), 1);
        assert!(matches!(
            &prog[0],
            TopLevel::Binding { name, .. } if name == "main"
        ));
    }

    #[test]
    fn test_parse_binop() {
        let e = parse_expr_str("2 + 3");
        assert!(matches!(
            e,
            Expr::BinOp { op: BinOpKind::Add, .. }
        ));
    }

    #[test]
    fn test_parse_precedence() {
        // 2 + 3 * 4 → Add(2, Mul(3, 4))
        let e = parse_expr_str("2 + 3 * 4");
        match e {
            Expr::BinOp {
                op: BinOpKind::Add,
                left,
                right,
            } => {
                assert!(matches!(*left, Expr::Lit(Literal::Int(2))));
                assert!(matches!(*right, Expr::BinOp { op: BinOpKind::Mul, .. }));
            }
            _ => panic!("unexpected: {:?}", e),
        }
    }

    #[test]
    fn test_parse_power_rtl() {
        // 2 ^ 3 ^ 4 → Pow(2, Pow(3, 4))
        let e = parse_expr_str("2 ^ 3 ^ 4");
        match e {
            Expr::BinOp {
                op: BinOpKind::Pow,
                right,
                ..
            } => {
                assert!(matches!(*right, Expr::BinOp { op: BinOpKind::Pow, .. }));
            }
            _ => panic!("unexpected: {:?}", e),
        }
    }

    #[test]
    fn test_parse_fn_def() {
        let prog = parse_str("f x = x + 1");
        match &prog[0] {
            TopLevel::Binding { name, params, .. } => {
                assert_eq!(name, "f");
                assert_eq!(params, &["x"]);
            }
            _ => panic!(),
        }
    }

    #[test]
    fn test_parse_app() {
        let e = parse_expr_str("f 3");
        assert!(matches!(e, Expr::App(_, _)));
    }

    #[test]
    fn test_parse_let() {
        let e = parse_expr_str("let x = 3 in x + 1");
        assert!(matches!(e, Expr::Let { .. }));
    }

    #[test]
    fn test_parse_if() {
        let e = parse_expr_str("if true then 1 else 0");
        assert!(matches!(e, Expr::If { .. }));
    }

    #[test]
    fn test_parse_tensor_1d() {
        let e = parse_expr_str("[1.0, 2.0, 3.0]");
        match e {
            Expr::TensorLit(rows) => {
                assert_eq!(rows.len(), 1);
                assert_eq!(rows[0].len(), 3);
            }
            _ => panic!(),
        }
    }

    #[test]
    fn test_parse_tensor_2d() {
        let e = parse_expr_str("[1.0, 2.0; 3.0, 4.0]");
        match e {
            Expr::TensorLit(rows) => {
                assert_eq!(rows.len(), 2);
                assert_eq!(rows[0].len(), 2);
                assert_eq!(rows[1].len(), 2);
            }
            _ => panic!(),
        }
    }

    #[test]
    fn test_parse_pipe() {
        let e = parse_expr_str("[1.0] |> sum");
        assert!(matches!(e, Expr::Pipe(_, _)));
    }

    #[test]
    fn test_parse_type_annotation() {
        let prog = parse_str("f : Int -> Int");
        assert!(matches!(&prog[0], TopLevel::TypeAnnotation { name, .. } if name == "f"));
    }

    #[test]
    fn test_parse_multiline() {
        let src = "x = 1\ny = 2\nmain = x + y";
        let prog = parse_str(src);
        assert_eq!(prog.len(), 3);
    }
}
