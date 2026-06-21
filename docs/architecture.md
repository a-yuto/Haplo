# Haplo — アーキテクチャ詳細

## ファイル構成

```
Haplo/
├── Cargo.toml          # 依存: ndarray = "0.16" のみ
├── SPEC.md             # 言語仕様インデックス
├── CLAUDE.md           # このガイドのインデックス
├── docs/               # 詳細ドキュメント
│   ├── spec-goals.md   # §0〜§2, §4, §5
│   ├── spec-syntax.md  # §3.1〜§3.5
│   ├── spec-types.md   # §3.6〜§3.8（北極星サンプル）
│   ├── spec-roadmap.md # §6〜付録
│   ├── architecture.md # このファイル
│   └── p1-plan.md      # P1 実装計画・未完成箇所
└── src/
    ├── ast.rs          # AST 型定義（ロジックなし）
    ├── lexer.rs        # トークナイザ
    ├── parser.rs       # 再帰下降パーサ
    ├── value.rs        # Value / Env / EvalError
    ├── interpreter.rs  # ツリーウォーキング評価器
    └── main.rs         # CLI エントリポイント + run()
```

パイプライン: `lex()` → `parse()` → `eval_program()` → `println!`

---

## 評価器（interpreter.rs）

**ツリーウォーキング**を選んだ理由: P1 の autodiff テープを `eval()` の呼び出し順に記録できるため。バイトコード方式では eval の中間状態にフックを挟みにくい。

- 組み込み関数（`sum`, `mean` 等）は `Value::Builtin` として環境に注入。`eval()` 内で名前を特別扱いしない。これにより |> パイプでも通常どおり使える
- 多引数関数は `desugar_lambda` でカリー化（`f x y = body` → `Lambda{x, Lambda{y, body}}`）。`rev().fold()` の順序に注意（rev なしだと引数順が逆になる）
- グローバル環境はファイルの記述順に構築（**前方参照不可**、P1 で two-pass 対応予定）

---

## 環境（value.rs）

- `Env` は Rc 永続連結リスト。`extend()` は O(1)、複数クロージャが同じ親 env を安全に共有できる
- `Value::Int` と `Value::Float` を分けて保持（整数除算の維持と表示の自然さのため。`6.0` → `"6.0"` と表示）
- テンソルは `Rc<ArrayD<f64>>` でくるむ（クロージャキャプチャ時のコピーを O(1) に抑える）。Arc ではなく Rc を使うのは P0 がシングルスレッドだから

---

## レキサー（lexer.rs）

- `Vec<char>` で保持する理由: `peek2()`（2文字先読み）に `chars[pos+1]` が必要なため
- 改行の意味判定: `open_depth == 0 && !last_was_continuation` のときのみ `Newline` を emit
  - `open_depth`: `(` / `[` の入れ子深度。括弧内は常に行継続
  - `last_was_continuation`: 演算子・コンマ・開き括弧の後は true。行がまだ続くことを示す
- `sum`/`mean` 等の組み込み名はキーワードにしない（`Ident` として通過させ、環境で解決）。将来ユーザが同名の関数を定義できるようにするため

---

## パーサー（parser.rs）

- **手書き再帰下降**（nom/pest/lalrpop 不採用）。エラーメッセージのカスタマイズと、Haplo の文法が小さいことが理由
- 演算子優先順位は**関数呼び出しチェーン**で実装（Pratt パーサ不採用）。`parse_additive` が `parse_matmul` を呼ぶ等、各優先順位レベルが独立した関数として見える
- `next_starts_atom` から `Minus` を除外（`10 - 3` を `App(10, -3)` に誤解釈しないため。負数引数は `f (-3)` と書く）
- `^` は再帰呼び出しで右結合を実現（`2^3^4` → `Pow(2, Pow(3, 4))`）
