# 数値計算DSL 要件定義書 兼 言語仕様（v2）

Haplo — 機械学習向けの純粋関数型・静的型付き DSL。
文法は Elm 風。テンソル演算は ndarray に委譲し、autodiff テープと依存型を Rust で自作する。

## 目次

| 文書 | 内容 |
|------|------|
| [docs/spec-goals.md](docs/spec-goals.md) | §0 言語哲学、§1 ターゲット、§2 機能要件、§4 非機能要件、§5 アーキテクチャ概観 |
| [docs/spec-syntax.md](docs/spec-syntax.md) | §3.1 字句、§3.2 テンソルリテラル、§3.3 束縛・関数定義、§3.4 演算子、§3.5 制御・反復 |
| [docs/spec-types.md](docs/spec-types.md) | §3.6 型の文法（依存型・shape 算術）、§3.7 プログラム構造、§3.8 北極星サンプル（線形回帰） |
| [docs/spec-roadmap.md](docs/spec-roadmap.md) | §6 開発計画、§7 マイルストーン G0〜G4、§8 技術的リスク、§9 未決定論点、付録 |

## 早見き

- **今どこ？** → P0 完了（G0/G1 達成）。次は P1（autodiff + `grad` + `iterate`）
- **北極星プログラム** → [docs/spec-types.md §3.8](docs/spec-types.md)（線形回帰サンプル）
- **実装計画** → [docs/spec-roadmap.md §6〜§7](docs/spec-roadmap.md)
