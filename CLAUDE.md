# Haplo — Claude Code 向けプロジェクトガイド

## プロジェクト概要

Haplo は機械学習向けの純粋関数型・静的型付き DSL。
テンソル演算（ndarray）と逆伝播 autodiff を Rust で実装する。
文法は Elm 風。言語仕様の全体像は `SPEC.md` を参照。

**現在のフェーズ: P0 完了（G0/G1 達成済み）**
次のターゲット: **P1 — autodiff テープ + `grad` + `iterate`（G2/G3）**

---

## ビルド・テスト

```bash
cargo build          # ビルド
cargo test           # 全テスト（57本）
cargo run -- foo.hpl # ファイル実行
```

テストは各ソースファイル末尾の `#[cfg(test)]` ブロックにある。
インタプリタのテストが最も多い（`g0_*` がスカラー15本、`g1_*` がテンソル10本）。

---

## ファイル構成

```
Haplo/
├── Cargo.toml          # 依存: ndarray = "0.16" のみ
├── SPEC.md             # 要件定義書 兼 言語仕様（v2）
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

## アーキテクチャの要点

### 評価器（interpreter.rs）
- **ツリーウォーキング**を選んだ理由: P1 の autodiff テープを `eval()` の呼び出し順に記録できるため
- 組み込み関数（`sum`, `mean` 等）は `Value::Builtin` として環境に注入する。`eval()` 内で名前を特別扱いしない
- 多引数関数は `desugar_lambda` でカリー化（`f x y = body` → `Lambda{x, Lambda{y, body}}`）
- グローバル環境はファイルの記述順に構築される（**前方参照不可**、P1 で two-pass 対応予定）

### 環境（value.rs）
- `Env` は Rc 永続連結リスト。`extend()` は O(1)、クロージャが安全に共有できる
- `Value::Int` と `Value::Float` を分けて保持（整数除算の維持と表示の自然さのため）
- テンソルは `Rc<ArrayD<f64>>` でくるむ（クロージャキャプチャ時のコピーを O(1) に抑える）

### レキサー（lexer.rs）
- 改行の意味判定: `open_depth == 0 && !last_was_continuation` のときのみ `Newline` を emit
- `sum`/`mean` 等の組み込み名はキーワードにしない（`Ident` として通過させ、環境で解決）

### パーサー（parser.rs）
- 手書き再帰下降（nom/pest/lalrpop 不採用）
- 演算子優先順位は関数呼び出しチェーンで実装（Pratt パーサ不採用）
- `next_starts_atom` から `Minus` を除外（`10 - 3` を `App(10, -3)` に誤解釈しないため）
- `^` は再帰呼び出しで右結合を実現

---

## 開発フェーズと次のステップ

| フェーズ | 状態 | 主な作業 |
|---------|------|---------|
| P0 | **完了** | lexer / parser / インタプリタ（スカラー+テンソル） |
| P1 | **次** | autodiff テープ、`grad`、`iterate`、前方参照 |
| P2 | 未着手 | 静的 shape 検査（固定次元） |
| P3 | 未着手 | 次元変数の単一化・shape 算術 |
| P4 | 未着手 | 完全な dependent 型 |

**P1 で実装する最小セット（G3 達成に必要）:**
- `Tape` 構造体（演算の記録）と逆伝播
- `grad : (Tensor -> f32) -> Tensor -> Tensor` 組み込み
- `iterate : a -> Int -> (a -> a) -> a` 組み込み（または再帰で代替）
- トップレベル定義の前方参照（two-pass スコープ構築）

---

## 未完成の箇所（P0 スコープ外）

- `reshape`: 第2引数（shape）の処理がダミー（`__reshape_applied__` は未定義）
- `case` 式: AST に未定義
- `fold`: 未実装
- `iterate`: 未実装（現在は再帰で代替可）
- 前方参照・相互再帰: 未対応

---

## コーディング規約

- コメントは `//`（非公開コード）または `///`（公開 API）
- テストブロック `#[cfg(test)]` へのコメントは不要
- 外部クレートは最小限（現在 `ndarray` のみ）
- `unwrap()` はテスト内と「到達不能」が証明できる箇所のみ許容
