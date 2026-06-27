# Haplo

機械学習向けの**純粋関数型・静的型付き DSL**。Rust 実装。

テンソルの shape（次元）を型レベルで検査し、行列積の内次元不一致や shape 違いを**実行前**に弾く。
autodiff（リバースモード）を内蔵し、`grad` で勾配関数を得られる。

```
-- 線形回帰の学習ループ（1ファイルで完結）
x = [1.0, 2.0, 3.0; 4.0, 5.0, 6.0; 7.0, 8.0, 9.0; 1.0, 0.0, 1.0]
y = [1.0, 2.0, 3.0, 0.5]
lr = 0.01

predict feats w b = feats @ w + b
mse pred target   = mean ((pred - target) ^ 2)
loss w            = mse (predict x w 0.0) y
step w            = w - lr * grad loss w

main = iterate (zeros [3]) 1000 step
```

## 特徴

| 機能 | 状態 |
|------|------|
| テンソル四則・行列積・転置・reshape | ✅ |
| リバースモード autodiff（`grad`） | ✅ |
| Shape の静的検査（固定次元・次元変数・shape 算術） | ✅ P3–P4 |
| 値依存 shape（`zeros [3]` → `Tensor[3]`） | ✅ P5 |
| 標準ライブラリ（abs / concat / flatten / norm / clip …） | ✅ P6 |
| 型注釈（`f : Tensor[m] -> Tensor[m*2]`） | ✅ |

## 使い方

```bash
# ビルド
cargo build

# ファイル実行
cargo run -- examples/linreg_train.hpl

# 全テスト（106本）
cargo test
```

## サンプルプログラム

| ファイル | 内容 |
|----------|------|
| `examples/linreg_train.hpl` | 線形回帰の学習ループ（北極星プログラム） |
| `examples/type_check.hpl` | 型注釈駆動の shape 検査ショーケース |
| `examples/stdlib_showcase.hpl` | 標準ライブラリ（P6 追加組み込み）の使用例 |
| `examples/linreg_forward.hpl` | 線形回帰の前向き計算のみ |
| `examples/activations.hpl` | 活性化関数のサンプル |
| `examples/functional.hpl` | 高階関数・部分適用のサンプル |

## 言語の概要

文法は Elm 風（ASCII のみ）。副作用なしの純粋関数型。

```
-- 変数束縛
x = 42.0

-- 関数定義（= で束縛、引数はスペース区切り）
add a b = a + b

-- 型注釈（オプション、shape 検査に使われる）
dot : Tensor[n] -> Tensor[n] -> f32
dot u v = sum (u * v)

-- 行列積
result = a @ b

-- 自動微分
df = grad loss w   -- loss の w に関する勾配

-- 反復
trained = iterate w0 1000 step
```

### 組み込み関数

| 関数 | 説明 |
|------|------|
| `zeros [n]` / `ones [n]` | 指定 shape のゼロ/一テンソル |
| `reshape t [m, n]` | テンソルを reshape |
| `transpose t` | 転置 |
| `sum` / `mean` / `max` / `min` | 集約 |
| `exp` / `log` / `tanh` / `softmax` | 数学関数 |
| `abs` / `clip lo hi` | 値域操作 |
| `concat u v` | 1D テンソルの連結（`Tensor[m+n]`） |
| `flatten t` | 2D → 1D に展開（`Tensor[m*n]`） |
| `norm t` | L2 ノルム |
| `grad f` | `f` の勾配関数を返す |
| `iterate init n step` | `step` を `n` 回適用 |

### Shape 検査の例

```
-- 内次元不一致はコンパイル時に弾かれる
a = [1.0, 2.0; 3.0, 4.0]   -- Tensor[2, 2]
b = [1.0, 2.0, 3.0]         -- Tensor[3]
main = a @ b                 -- error: shape mismatch (2 vs 3)

-- 型注釈と実際の shape が合わなければエラー
w : Tensor[3, 3]
w = [1.0, 0.0; 0.0, 1.0]   -- error: annotation mismatch
```

## アーキテクチャ

```
ソース → lexer → parser → AST
                            │
                    shape staging（抽象評価・P2）
                            │
                    型検査・shape 検査（P3–P5）
                            │
                    評価器（インタプリタ）
                     ├─ ndarray（配列演算）
                     └─ autodiff テープ（逆伝播）
```

詳細は [docs/architecture.md](docs/architecture.md) を参照。

## ドキュメント

| 文書 | 内容 |
|------|------|
| [SPEC.md](SPEC.md) | 言語仕様インデックス |
| [docs/spec-goals.md](docs/spec-goals.md) | 言語哲学・ターゲット・機能要件 |
| [docs/spec-syntax.md](docs/spec-syntax.md) | 字句・リテラル・演算子・制御 |
| [docs/spec-types.md](docs/spec-types.md) | 型文法・依存型・shape 算術 |
| [docs/spec-roadmap.md](docs/spec-roadmap.md) | 開発計画・マイルストーン |
| [docs/architecture.md](docs/architecture.md) | ファイル構成・設計の選択 |

## 現在のフェーズ

**P6 完了**（標準ライブラリ拡充・concat/flatten の shape 推論）。

次フェーズ **P7**：レイアウト厳密化・REPL・`import`。
