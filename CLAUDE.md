# Haplo — Claude Code 向けプロジェクトガイド

Haplo は機械学習向けの純粋関数型・静的型付き DSL（Rust 実装）。
**現在のフェーズ: P1 完了（G3 達成済み — autodiff + `grad` + `iterate` + 前方参照）。次: P2 — 静的 shape 検査**

## ビルド・テスト

```bash
cargo build          # ビルド
cargo test           # 全テスト（75本）
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
