# Haplo — Claude Code 向けプロジェクトガイド

Haplo は機械学習向けの純粋関数型・静的型付き DSL（Rust 実装）。
**現在のフェーズ: P2 完了（G4 達成済み — shape staging パスで行列積・要素ごと演算の不整合を実行前に検出）。次: P3 — 次元変数の単一化・shape 算術**

## ビルド・テスト

```bash
cargo build          # ビルド
cargo test           # 全テスト（90本）
cargo run -- foo.hpl # ファイル実行
cargo run -- examples/linreg_train.hpl  # 北極星プログラム（線形回帰の学習）
```

## ドキュメント

| 文書 | 内容 |
|------|------|
| [SPEC.md](SPEC.md) | 言語仕様インデックス |
| [docs/spec-goals.md](docs/spec-goals.md) | 言語哲学・ターゲット・機能要件・アーキテクチャ概観 |
| [docs/spec-syntax.md](docs/spec-syntax.md) | 字句・テンソルリテラル・束縛・演算子・制御 |
| [docs/spec-types.md](docs/spec-types.md) | 型の文法・依存型・北極星サンプル（線形回帰） |
| [docs/spec-roadmap.md](docs/spec-roadmap.md) | 開発計画・マイルストーン・リスク |
| [docs/architecture.md](docs/architecture.md) | ファイル構成・評価器/環境/レキサー/パーサの設計選択 |
| [docs/p1-plan.md](docs/p1-plan.md) | P1 実装計画・未完成箇所・コーディング規約 |

## 読む順

1. 全体把握 → `SPEC.md`（インデックス）
2. 今すぐ実装 → `docs/p1-plan.md`（何が残っているか）
3. 設計の理由 → `docs/architecture.md`
4. 文法の詳細 → `docs/spec-syntax.md` / `docs/spec-types.md`

## コメント方針

**コメントは可能な限り記載する**（テストコードを含む）。コードが「何をしているか」だけでなく、
**「なぜそうしているか」**を残すことを重視する。具体的には:

- **どのような実装をしているか**：処理の意図・アルゴリズム・データの流れを説明する。
  自明な1行（`i += 1` 等）への注釈は不要だが、まとまった処理の塊には目的を書く。
- **なぜ今の実装を選んだか**：複数ありうる設計のうち現在のものを選んだ理由と、
  検討して**採用しなかった代替案**（および却下理由）を書く。
  例: 「ツリーウォーキングを選んだ理由は eval の呼び出し順にテープを記録できるから」
  「`&mut Tape` を全段に通す案より thread_local の方が変更が小さい」など。
- **数式・不変条件**：autodiff の局所微分や逆伝播ループの不変条件のように、
  後から検証しづらいロジックは導出や前提条件を明記する。
- **テストコメント**：各テストが「何を・なぜ検証するか」、期待値の根拠（数学的導出・境界条件）を書く。

書式は `//`（非公開コード）/ `///`（公開 API）。設計判断の背景は `docs/architecture.md` に集約し、
コード側コメントと相互に補完させる。
