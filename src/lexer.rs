// Haplo のレキサー（字句解析器）。
// ソーステキストを読んで Token の列に変換する。パーサはこのトークン列だけを見る。
//
// 設計上の重要な選択:
// 1. 改行を「意味ある区切り」と「単なる空白」に振り分ける処理をここで行う。
//    パーサ側で改行をケアしなくて済むようにするため。
// 2. sum/mean などの組み込み関数名はキーワードにしない（Ident として通過させる）。
//    将来ユーザが同名の関数を定義したり、シャドーイングできる余地を残すため。

#[derive(Debug, Clone, PartialEq)]
pub enum TokenKind {
    // リテラル
    Int(i64),
    Float(f64),
    Bool(bool),

    // 識別子
    Ident(String),

    // キーワード
    Let,
    In,
    If,
    Then,
    Else,

    // 演算子
    Plus,
    Minus,
    Star,
    Slash,
    Caret,
    At,
    EqEq,
    BangEq,
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Colon,
    Arrow,
    Comma,
    Semicolon,
    Pipe, // |>

    // グルーピング
    LParen,
    RParen,
    LBrack,
    RBrack,

    // レイアウト
    Newline,
    Eof,
}

#[derive(Debug, Clone)]
pub struct Span {
    pub line: u32,
    pub col: u32,
}

#[derive(Debug, Clone)]
pub struct Token {
    pub kind: TokenKind,
    pub span: Span,
}

#[derive(Debug)]
pub enum LexError {
    UnexpectedChar {
        ch: char,
        line: u32,
        col: u32,
    },
    UnterminatedBlockComment {
        line: u32,
        col: u32,
    },
    MalformedNumber {
        raw: String,
        line: u32,
        col: u32,
    },
}

impl std::fmt::Display for LexError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LexError::UnexpectedChar { ch, line, col } => {
                write!(f, "予期しない文字 {:?} ({}:{})", ch, line, col)
            }
            LexError::UnterminatedBlockComment { line, col } => {
                write!(f, "ブロックコメントが閉じられていません ({}:{})", line, col)
            }
            LexError::MalformedNumber { raw, line, col } => {
                write!(f, "不正な数値リテラル {:?} ({}:{})", raw, line, col)
            }
        }
    }
}

// ソーステキストをあらかじめ Vec<char> に変換して保持する。
// 理由: peek2()（2文字先読み）が必要なため。
// Chars イテレータはインデックスアクセスできないので、
// Vec に変換することで chars[pos+1] という直接アクセスが可能になる。
// メモリ: 入力が大きい場合は問題になりうるが、DSL のソースは通常小さいので許容。
struct Lexer {
    chars: Vec<char>,
    pos: usize,
    line: u32,
    col: u32,
    // `(` / `[` の入れ子深度を追跡するカウンタ。
    // 括弧の中では改行を無視する（行継続とみなす）ため。
    // 例: [1.0,\n 2.0] の改行はトークンに変換しない。
    // 代替: ( と [ を別々にカウントすることもできるが、
    // ネストが正しく対応する限り1つのカウンタで十分なのでシンプルさを優先した。
    open_depth: i32,
    // 直前に emit したトークンが「式の途中」を示す場合 true にするフラグ。
    // 演算子・コンマ・開き括弧・キーワード（let/if等）の後は、
    // 式がまだ続くので次の改行を区切りとして扱わない。
    //
    // Haskell や Python のようなオフサイドルール（インデント量で構造を決める）は
    // 実装が複雑になりすぎるため採用しなかった。
    // 代わりに「前のトークンで判断する」シンプルな後ろ向きルールを使う。
    last_was_continuation: bool,
}

impl Lexer {
    fn new(source: &str) -> Self {
        Lexer {
            chars: source.chars().collect(),
            pos: 0,
            line: 1,
            col: 1,
            open_depth: 0,
            last_was_continuation: true, // ファイル先頭の改行は無視
        }
    }

    fn peek(&self) -> Option<char> {
        self.chars.get(self.pos).copied()
    }

    fn peek2(&self) -> Option<char> {
        self.chars.get(self.pos + 1).copied()
    }

    fn advance(&mut self) -> Option<char> {
        let ch = self.chars.get(self.pos).copied()?;
        self.pos += 1;
        if ch == '\n' {
            self.line += 1;
            self.col = 1;
        } else {
            self.col += 1;
        }
        Some(ch)
    }

    fn span(&self) -> Span {
        Span {
            line: self.line,
            col: self.col,
        }
    }

    fn skip_line_comment(&mut self) {
        while let Some(ch) = self.peek() {
            if ch == '\n' {
                break;
            }
            self.advance();
        }
    }

    // ネスト可能なブロックコメント {- ... -} をスキップする。
    // depth カウンタで {- と -} の対応を追跡し、depth が 0 になったら終了。
    // Elm や Haskell と同様にネストを許可することで、
    // コメントアウト時にすでにコメントが含まれていてもエラーにならない。
    fn skip_block_comment(&mut self, start_line: u32, start_col: u32) -> Result<(), LexError> {
        // `{-` は消費済み
        let mut depth = 1i32;
        while depth > 0 {
            match self.peek() {
                None => {
                    return Err(LexError::UnterminatedBlockComment {
                        line: start_line,
                        col: start_col,
                    })
                }
                Some('{') if self.peek2() == Some('-') => {
                    self.advance();
                    self.advance();
                    depth += 1;
                }
                Some('-') if self.peek2() == Some('}') => {
                    self.advance();
                    self.advance();
                    depth -= 1;
                }
                _ => {
                    self.advance();
                }
            }
        }
        Ok(())
    }

    // 数値リテラルをスキャンして Int または Float トークンを返す。
    // 判定ロジック:
    //   - 小数点の後に数字が続く → Float（例: 3.14）
    //   - e/E が続く → 指数表記の Float（例: 1e-3）
    //   - それ以外 → Int
    // アンダースコアを区切り文字として許可（例: 1_000_000）し、
    // パース前にフィルタリングして除去する。
    // 代替: lexer で Float と Int を統一して後で型推論で決める方法もあるが、
    // リテラルの見た目（3 vs 3.0）でユーザの意図が明確なのでここで分ける。
    fn scan_number(&mut self, first: char) -> Result<TokenKind, LexError> {
        let start_line = self.line;
        let start_col = self.col - 1;
        let mut raw = String::new();
        raw.push(first);

        while let Some(c) = self.peek() {
            if c.is_ascii_digit() || c == '_' {
                raw.push(c);
                self.advance();
            } else {
                break;
            }
        }

        let is_float = self.peek() == Some('.')
            && self.peek2().map_or(false, |c| c.is_ascii_digit());
        let has_exp = !is_float
            && (self.peek() == Some('e') || self.peek() == Some('E'));

        if is_float {
            raw.push('.');
            self.advance();
            while let Some(c) = self.peek() {
                if c.is_ascii_digit() {
                    raw.push(c);
                    self.advance();
                } else {
                    break;
                }
            }
        }

        if is_float || has_exp {
            if self.peek() == Some('e') || self.peek() == Some('E') {
                raw.push(self.advance().unwrap());
                if self.peek() == Some('+') || self.peek() == Some('-') {
                    raw.push(self.advance().unwrap());
                }
                while let Some(c) = self.peek() {
                    if c.is_ascii_digit() {
                        raw.push(c);
                        self.advance();
                    } else {
                        break;
                    }
                }
            }
            let clean: String = raw.chars().filter(|c| *c != '_').collect();
            clean.parse::<f64>().map(TokenKind::Float).map_err(|_| {
                LexError::MalformedNumber {
                    raw,
                    line: start_line,
                    col: start_col,
                }
            })
        } else {
            let clean: String = raw.chars().filter(|c| *c != '_').collect();
            clean.parse::<i64>().map(TokenKind::Int).map_err(|_| {
                LexError::MalformedNumber {
                    raw,
                    line: start_line,
                    col: start_col,
                }
            })
        }
    }

    // 識別子をスキャンし、末尾の ' (プライム) を1文字だけ許容する。
    // x' のような表記は数学で「xの別バージョン」を表す慣習で、
    // Haskell 等でも使われる。ループ変数の更新値などに便利。
    // 複数プライム（x''）は現在非対応（1文字で十分な範囲を想定）。
    fn scan_ident(&mut self, first: char) -> String {
        let mut s = String::new();
        s.push(first);
        while let Some(c) = self.peek() {
            if c.is_alphanumeric() || c == '_' {
                s.push(c);
                self.advance();
            } else {
                break;
            }
        }
        // 末尾プライム
        if self.peek() == Some('\'') {
            s.push('\'');
            self.advance();
        }
        s
    }

    // 識別子文字列をキーワードか普通の識別子かに振り分ける。
    // sum/mean/exp などの組み込み関数名はここに加えない。
    // 理由: 組み込み関数はインタプリタの環境に Value::Builtin として注入するだけで、
    // レキサーレベルでキーワードにすると将来ユーザが同名の関数を定義できなくなる。
    fn keyword_or_ident(s: String) -> TokenKind {
        match s.as_str() {
            "true" => TokenKind::Bool(true),
            "false" => TokenKind::Bool(false),
            "let" => TokenKind::Let,
            "in" => TokenKind::In,
            "if" => TokenKind::If,
            "then" => TokenKind::Then,
            "else" => TokenKind::Else,
            _ => TokenKind::Ident(s),
        }
    }

    // トークンを1つ生成して返す（None は返さず Eof トークンで終端を示す）。
    // 改行文字 \n を検出したとき、以下の条件を両方満たす場合にのみ Newline トークンを emit する:
    //   1. 括弧の外（open_depth == 0）
    //   2. 直前が値を表すトークン（!last_was_continuation）
    // これにより:
    //   "2 + \n 3"  → Newline を emit しない（+ の後は継続）
    //   "2 \n 3"    → Newline を emit する（定義の区切り）
    //   "[1,\n 2]"  → Newline を emit しない（括弧内は常に継続）
    fn next_token(&mut self) -> Result<Option<Token>, LexError> {
        // 空白（改行以外）をスキップ
        loop {
            match self.peek() {
                Some(' ') | Some('\t') | Some('\r') => {
                    self.advance();
                }
                Some('\n') => {
                    let span = self.span();
                    self.advance();
                    // 意味のある改行かどうか判定
                    if self.open_depth == 0 && !self.last_was_continuation {
                        self.last_was_continuation = true;
                        return Ok(Some(Token {
                            kind: TokenKind::Newline,
                            span,
                        }));
                    }
                    // 継続扱い: 無視
                }
                Some('-') if self.peek2() == Some('-') => {
                    self.advance();
                    self.advance();
                    self.skip_line_comment();
                }
                Some('{') if self.peek2() == Some('-') => {
                    let l = self.line;
                    let c = self.col;
                    self.advance();
                    self.advance();
                    self.skip_block_comment(l, c)?;
                }
                _ => break,
            }
        }

        let span = self.span();
        let ch = match self.advance() {
            None => {
                return Ok(Some(Token {
                    kind: TokenKind::Eof,
                    span,
                }))
            }
            Some(c) => c,
        };

        let kind = match ch {
            '+' => {
                self.last_was_continuation = true;
                TokenKind::Plus
            }
            '-' if self.peek() == Some('>') => {
                self.advance();
                self.last_was_continuation = true;
                TokenKind::Arrow
            }
            '-' => {
                self.last_was_continuation = true;
                TokenKind::Minus
            }
            '*' => {
                self.last_was_continuation = true;
                TokenKind::Star
            }
            '/' => {
                self.last_was_continuation = true;
                TokenKind::Slash
            }
            '^' => {
                self.last_was_continuation = true;
                TokenKind::Caret
            }
            '@' => {
                self.last_was_continuation = true;
                TokenKind::At
            }
            '=' if self.peek() == Some('=') => {
                self.advance();
                self.last_was_continuation = true;
                TokenKind::EqEq
            }
            '=' => {
                self.last_was_continuation = true;
                TokenKind::Eq
            }
            '!' if self.peek() == Some('=') => {
                self.advance();
                self.last_was_continuation = true;
                TokenKind::BangEq
            }
            '<' if self.peek() == Some('=') => {
                self.advance();
                self.last_was_continuation = true;
                TokenKind::Le
            }
            '<' => {
                self.last_was_continuation = true;
                TokenKind::Lt
            }
            '>' if self.peek() == Some('=') => {
                self.advance();
                self.last_was_continuation = true;
                TokenKind::Ge
            }
            '>' => {
                self.last_was_continuation = true;
                TokenKind::Gt
            }
            ':' => {
                self.last_was_continuation = true;
                TokenKind::Colon
            }
            ',' => {
                self.last_was_continuation = true;
                TokenKind::Comma
            }
            ';' => {
                self.last_was_continuation = true;
                TokenKind::Semicolon
            }
            '|' if self.peek() == Some('>') => {
                self.advance();
                self.last_was_continuation = true;
                TokenKind::Pipe
            }
            '(' => {
                self.open_depth += 1;
                self.last_was_continuation = true;
                TokenKind::LParen
            }
            ')' => {
                self.open_depth -= 1;
                self.last_was_continuation = false;
                TokenKind::RParen
            }
            '[' => {
                self.open_depth += 1;
                self.last_was_continuation = true;
                TokenKind::LBrack
            }
            ']' => {
                self.open_depth -= 1;
                self.last_was_continuation = false;
                TokenKind::RBrack
            }
            c if c.is_ascii_digit() => {
                self.last_was_continuation = false;
                self.scan_number(c)?
            }
            c if c.is_alphabetic() || c == '_' => {
                let ident = self.scan_ident(c);
                let kind = Self::keyword_or_ident(ident);
                // キーワードは継続扱い（`let`, `if`, `then`, `else`、`in`）
                self.last_was_continuation = matches!(
                    kind,
                    TokenKind::Let
                        | TokenKind::In
                        | TokenKind::If
                        | TokenKind::Then
                        | TokenKind::Else
                        | TokenKind::Eq
                );
                kind
            }
            c => {
                return Err(LexError::UnexpectedChar {
                    ch: c,
                    line: span.line,
                    col: span.col,
                })
            }
        };

        // Bool / Ident は継続しない（値を表す）
        match &kind {
            TokenKind::Bool(_) | TokenKind::Ident(_) => {
                self.last_was_continuation = false;
            }
            _ => {}
        }

        Ok(Some(Token { kind, span }))
    }
}

pub fn lex(source: &str) -> Result<Vec<Token>, LexError> {
    let mut lexer = Lexer::new(source);
    let mut tokens = Vec::new();
    loop {
        match lexer.next_token()? {
            None => break,
            Some(tok) => {
                let is_eof = matches!(tok.kind, TokenKind::Eof);
                tokens.push(tok);
                if is_eof {
                    break;
                }
            }
        }
    }
    Ok(tokens)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kinds(src: &str) -> Vec<TokenKind> {
        lex(src).unwrap().into_iter().map(|t| t.kind).collect()
    }

    #[test]
    fn test_lex_integer() {
        assert_eq!(kinds("42"), vec![TokenKind::Int(42), TokenKind::Eof]);
    }

    #[test]
    fn test_lex_float() {
        let k = kinds("3.14");
        assert!(matches!(k[0], TokenKind::Float(x) if (x - 3.14).abs() < 1e-9));
        assert_eq!(k[1], TokenKind::Eof);
    }

    #[test]
    fn test_lex_float_exp() {
        let k = kinds("1e-3");
        assert!(matches!(k[0], TokenKind::Float(x) if (x - 0.001).abs() < 1e-9));
    }

    #[test]
    fn test_lex_bool() {
        assert_eq!(
            kinds("true false"),
            vec![
                TokenKind::Bool(true),
                TokenKind::Bool(false),
                TokenKind::Eof
            ]
        );
    }

    #[test]
    fn test_lex_operators() {
        assert_eq!(
            kinds("2 + 3"),
            vec![
                TokenKind::Int(2),
                TokenKind::Plus,
                TokenKind::Int(3),
                TokenKind::Eof
            ]
        );
    }

    #[test]
    fn test_lex_line_comment() {
        assert_eq!(kinds("-- hi\n42"), vec![TokenKind::Int(42), TokenKind::Eof]);
    }

    #[test]
    fn test_lex_block_comment() {
        assert_eq!(kinds("{- x -}42"), vec![TokenKind::Int(42), TokenKind::Eof]);
    }

    #[test]
    fn test_lex_nested_block_comment() {
        assert_eq!(
            kinds("{- {- nested -} -}42"),
            vec![TokenKind::Int(42), TokenKind::Eof]
        );
    }

    #[test]
    fn test_lex_newline_significant() {
        assert_eq!(
            kinds("2\n3"),
            vec![
                TokenKind::Int(2),
                TokenKind::Newline,
                TokenKind::Int(3),
                TokenKind::Eof
            ]
        );
    }

    #[test]
    fn test_lex_newline_after_operator() {
        // 演算子の後の改行は継続扱い
        assert_eq!(
            kinds("2 +\n3"),
            vec![
                TokenKind::Int(2),
                TokenKind::Plus,
                TokenKind::Int(3),
                TokenKind::Eof
            ]
        );
    }

    #[test]
    fn test_lex_newline_inside_brackets() {
        // 括弧内の改行は継続扱い
        assert_eq!(
            kinds("(2\n+\n3)"),
            vec![
                TokenKind::LParen,
                TokenKind::Int(2),
                TokenKind::Plus,
                TokenKind::Int(3),
                TokenKind::RParen,
                TokenKind::Eof
            ]
        );
    }

    #[test]
    fn test_lex_prime_ident() {
        assert_eq!(
            kinds("x'"),
            vec![TokenKind::Ident("x'".to_string()), TokenKind::Eof]
        );
    }

    #[test]
    fn test_lex_pipe() {
        assert_eq!(
            kinds("x |> f"),
            vec![
                TokenKind::Ident("x".to_string()),
                TokenKind::Pipe,
                TokenKind::Ident("f".to_string()),
                TokenKind::Eof
            ]
        );
    }

    #[test]
    fn test_lex_arrow() {
        assert_eq!(
            kinds("->"),
            vec![TokenKind::Arrow, TokenKind::Eof]
        );
    }
}
