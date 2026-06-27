// Haplo の再帰下降パーサ。
// Token の列（lexer の出力）を受け取り、Program（AST）に変換する。
//
// 設計の選択: 手書き再帰下降 vs パーサジェネレータ（nom, pest, lalrpop）
// 手書きを選んだ理由:
//   - エラーメッセージを自由にカスタマイズできる（「〇〇の後に = が必要です」等）
//   - Haplo の文法は小さく、パーサジェネレータの習得コストが見合わない
//   - 再帰下降はコードの流れが文法規則と1対1で対応するため読みやすい
//
// 演算子優先順位の実装: Pratt パーサ vs 関数の呼び出しチェーン
// 呼び出しチェーン（parse_additive が parse_matmul を呼ぶ等）を選んだ理由:
//   - Pratt パーサは汎用的だが、演算子が固定・少数の場合はオーバーキル
//   - チェーン方式はそれぞれの優先順位レベルが独立した関数として見える
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

// パーサの状態を保持する構造体。
// tokens: lexer が返したトークン列への参照（所有権は持たない）
// pos:    現在読んでいる位置。advance() で進む。
// Eof トークンの後に pos が進まないよう advance() 内でガードしている。
// これにより EOF をどこで検出してもパニックしない。
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

    // Newline トークンを読み飛ばすヘルパー。
    // キーワード（let, in, then, else）や `=` の後に呼ぶことで、
    // ユーザが式を次の行に書いても正しくパースできる。
    // 例: "let x =\n  3 in x" の \n をここで吸収する。
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

    // 型注釈（name : Type）と束縛（name params* = expr）を区別してパースする。
    // 識別子の後のトークンで判定:
    //   : → 型注釈
    //   識別子 or = → 束縛（識別子はパラメータ名）
    // この2種類だけが先頭にくるため、それ以外はエラー。
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
    // 型式 / 次元式（P4 shape 算術）
    // ----------------------------------------------------------------

    // テンソル型注釈の次元位置に書ける算術式をパースする（P4 新規）。
    // 文法: dim_expr = dim_term (('+' | '-') dim_term)*
    // 例: `m+n`, `m-1`, `m+n-1`
    // ',' や ']' は dim_expr の終端として扱われ、parse_dim_expr がそこで止まる。
    fn parse_dim_expr(&mut self) -> Result<DimExpr, ParseError> {
        let mut lhs = self.parse_dim_term()?;
        loop {
            match self.peek().clone() {
                TokenKind::Plus => {
                    self.advance();
                    let rhs = self.parse_dim_term()?;
                    lhs = DimExpr::Add(Box::new(lhs), Box::new(rhs));
                }
                TokenKind::Minus => {
                    self.advance();
                    let rhs = self.parse_dim_term()?;
                    lhs = DimExpr::Sub(Box::new(lhs), Box::new(rhs));
                }
                _ => break,
            }
        }
        Ok(lhs)
    }

    // 次元式の乗算レベル: dim_term = dim_atom ('*' dim_atom)*
    fn parse_dim_term(&mut self) -> Result<DimExpr, ParseError> {
        let mut lhs = self.parse_dim_atom()?;
        loop {
            if matches!(self.peek(), TokenKind::Star) {
                self.advance();
                let rhs = self.parse_dim_atom()?;
                lhs = DimExpr::Mul(Box::new(lhs), Box::new(rhs));
            } else {
                break;
            }
        }
        Ok(lhs)
    }

    // 次元式の原子: 整数リテラルまたは次元変数名
    fn parse_dim_atom(&mut self) -> Result<DimExpr, ParseError> {
        match self.peek().clone() {
            TokenKind::Int(n) => {
                self.advance();
                Ok(DimExpr::Lit(n as usize))
            }
            TokenKind::Ident(var) => {
                self.advance();
                Ok(DimExpr::Var(var))
            }
            _ => {
                let tok = self.peek_token();
                Err(ParseError::UnexpectedToken {
                    got: format!("{:?}", tok.kind),
                    expected: "次元（整数または識別子）",
                    span: tok.span.clone(),
                })
            }
        }
    }

    // `->` は右結合: `A -> B -> C` = `A -> (B -> C)` = Arrow(A, Arrow(B, C))。
    // 左結合（while ループ）にすると `Arrow(Arrow(A,B), C)` になり、
    // decompose_arrow が先頭 Arrow の lhs（= Arrow(A,B)）を「1つ目の引数型」として
    // shape_of_type に渡してしまい、複数引数の型注釈が機能しなくなる。
    // 右結合（再帰呼び出し）なら decompose_arrow がネストを正しく順番に剥がせる。
    fn parse_type_expr(&mut self) -> Result<TypeExpr, ParseError> {
        let t = self.parse_type_atom()?;
        if matches!(self.peek(), TokenKind::Arrow) {
            self.advance();
            // 再帰で右辺を右結合にパース（`A -> B -> C` = `A -> (B -> C)`）。
            let rhs = self.parse_type_expr()?;
            Ok(TypeExpr::Arrow(Box::new(t), Box::new(rhs)))
        } else {
            Ok(t)
        }
    }

    fn parse_type_atom(&mut self) -> Result<TypeExpr, ParseError> {
        match self.peek().clone() {
            TokenKind::Ident(name) => {
                self.advance();
                // "Tensor[m, n]" のような次元付きテンソル型を特別扱いでパースする。
                // 次元には整数リテラル（固定サイズ）または識別子（次元変数）を許可する。
                // 固定サイズは TypeDim::Fixed、次元変数は TypeDim::Var として名前ごと保持する。
                // P2 までは変数名を捨てていたが、P3 の固定次元検査・P4 の単一化で使うため残す。
                if name == "Tensor" && matches!(self.peek(), TokenKind::LBrack) {
                    self.advance(); // consume `[`
                    let mut dims = Vec::new();
                    loop {
                        // P4: 次元に算術式（m+n, m*n 等）を許可する。
                        // parse_dim_expr で DimExpr を組み立て、単純な Lit/Var は
                        // TypeDim::Fixed/Var に、複合式は TypeDim::Expr に畳む。
                        let expr = self.parse_dim_expr()?;
                        let dim = match expr {
                            DimExpr::Lit(n) => TypeDim::Fixed(n),
                            DimExpr::Var(s) => TypeDim::Var(s),
                            e => TypeDim::Expr(e),
                        };
                        dims.push(dim);
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

    // ^ 演算子を右結合でパースする。
    // 通常の左再帰ループではなく、再帰呼び出しで実装する。
    // "2 ^ 3 ^ 4" → Pow(2, Pow(3, 4)) となり、数学の慣習に合う。
    // 代替の左結合（Pow(Pow(2,3), 4)）は数学的に間違いなので採用しない。
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

    // 空白による関数適用をパースする（最高優先順位）。
    // アトムを1つパースした後、next_starts_atom が真の間はアトムを読み続け、
    // App(App(func, arg1), arg2) のように左再帰でネストする。
    // 例: "f a b" → App(App(Var"f", Var"a"), Var"b")
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

    // 次のトークンが「関数の引数として読める式の先頭か」を判定する。
    // `-` をここに含めない理由:
    //   "10 - 3" を parse_application が "App(10, -3)" と解釈するのを防ぐため。
    //   `-` を含めると、10 の後に `-3`（単項マイナス付きの 3）が来たと解釈され、
    //   `10` を関数として `-3` に適用しようとしてしまう。
    // 負数を関数の引数として渡す場合は括弧を使う: `f (-3)`
    // これは Elm や Haskell でも同様の制約がある。
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

    // テンソルリテラル [e1, e2; e3, e4] をパースする。
    // 要素式を parse_expr で評価し、カンマで区切って現在行に追加、
    // セミコロンで新しい行を開始、] で終了する。
    // 各要素は任意の式（変数・計算式を含む）にできる。
    // 例: [x + 1, y * 2] のような動的な値も有効。
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
    fn test_parse_tensor_type_dims() {
        // テンソル型の次元は、整数なら TypeDim::Fixed、識別子なら TypeDim::Var として
        // 名前ごと保持されることを確認する（P3 で固定次元検査に使うため変数名を捨てない）。
        // 戻り型は Arrow の右側に来るので、`Tensor[3, n]` を引数に持つ関数型で検査する。
        let prog = parse_str("g : Tensor[3, n] -> Tensor[n]");
        match &prog[0] {
            TopLevel::TypeAnnotation { ty, .. } => match ty {
                TypeExpr::Arrow(arg, _) => match arg.as_ref() {
                    TypeExpr::Tensor(dims) => {
                        assert_eq!(
                            dims,
                            &vec![TypeDim::Fixed(3), TypeDim::Var("n".to_string())]
                        );
                    }
                    other => panic!("Tensor 型を期待: {:?}", other),
                },
                other => panic!("Arrow 型を期待: {:?}", other),
            },
            other => panic!("TypeAnnotation を期待: {:?}", other),
        }
    }

    #[test]
    fn test_parse_multiline() {
        let src = "x = 1\ny = 2\nmain = x + y";
        let prog = parse_str(src);
        assert_eq!(prog.len(), 3);
    }
}
