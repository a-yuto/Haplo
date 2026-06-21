// reverse-mode 自動微分（autodiff）のテープ実装。
//
// 仕組み: 演算を実行するたびに「演算ノード」をテープ（Wengert list）へ順に積む。
// 各ノードは forward 計算済みの値と、親ノードへの参照を持つ Op を保持する。
// すべての演算が終わったら backward() で末尾（出力）から逆向きに随伴（adjoint）を
// 累積し、各ノードの勾配を求める。
//
// テープを thread_local に置く理由:
// ツリーウォーキング評価器の eval() は AST を深く再帰するため、&mut Tape を全段に
// 通すとシグネチャ変更が広範囲に及ぶ。thread_local なら eval() の形を変えずに、
// grad の評価中だけテープへ記録できる（docs/architecture.md の設計方針どおり）。
//
// スカラーは 0 次元の ArrayD で表す。これにより スカラーとテンソルを同じノード型で
// 扱え、要素ごと演算のブロードキャストも一様に書ける。

use ndarray::{ArrayD, Ix1, Ix2, IxDyn};
use std::cell::RefCell;

use crate::ast::BinOpKind;
use crate::value::EvalError;

// テープ上の1演算ノード。value は forward 計算の結果（backward で再利用する）。
struct Node {
    value: ArrayD<f64>,
    op: Op,
}

// 演算の種類。各バリアントが親ノードの index を保持する。
// Leaf は入力変数または定数（親を持たない）。
enum Op {
    Leaf,
    Add(usize, usize),
    Sub(usize, usize),
    Mul(usize, usize),
    Div(usize, usize),
    Pow(usize, usize),
    MatMul(usize, usize),
    Neg(usize),
    Exp(usize),
    Log(usize),
    Tanh(usize),
    Sqrt(usize),
    Sum(usize),
    Mean(usize),
}

struct Tape {
    nodes: Vec<Node>,
}

thread_local! {
    // テープのスタック。grad のネストにも備えてスタックにしているが、
    // 通常は深さ1（grad の中で grad を呼ばない）。grad スコープ外では空。
    static TAPES: RefCell<Vec<Tape>> = const { RefCell::new(Vec::new()) };
}

// 0 次元 ArrayD（スカラー）を作る。定数スカラーを leaf に持ち上げる際に使う。
pub fn scalar(x: f64) -> ArrayD<f64> {
    ArrayD::from_elem(IxDyn(&[]), x)
}

// 新しいテープスコープを開始する（grad の入口で呼ぶ）。
pub fn tape_begin() {
    TAPES.with(|t| t.borrow_mut().push(Tape { nodes: Vec::new() }));
}

// 現在のテープスコープを破棄する（grad の出口で呼ぶ）。
pub fn tape_end() {
    TAPES.with(|t| {
        t.borrow_mut().pop();
    });
}

// 現在のテープ（スタック先頭）に対して f を実行するヘルパ。
// テープが無いときに呼ぶのは内部バグ（routing 側で tape_active を確認済み）。
fn with_top<R>(f: impl FnOnce(&mut Tape) -> R) -> R {
    TAPES.with(|t| {
        let mut stack = t.borrow_mut();
        let top = stack
            .last_mut()
            .expect("autodiff: アクティブなテープがありません");
        f(top)
    })
}

// 葉ノード（入力変数または定数）を積んで index を返す。
pub fn leaf(value: ArrayD<f64>) -> usize {
    with_top(|t| {
        let id = t.nodes.len();
        t.nodes.push(Node {
            value,
            op: Op::Leaf,
        });
        id
    })
}

// 二項演算ノードを積む。forward 値を計算し、対応する Op を記録する。
// 要素ごと演算（Add/Sub/Mul/Div/Pow）は scalar↔tensor のブロードキャストに対応。
pub fn binop(op: &BinOpKind, a: usize, b: usize) -> Result<usize, EvalError> {
    with_top(|t| {
        let (val, node_op) = {
            let av = &t.nodes[a].value;
            let bv = &t.nodes[b].value;
            match op {
                BinOpKind::Add => (ew(av, bv, |x, y| x + y), Op::Add(a, b)),
                BinOpKind::Sub => (ew(av, bv, |x, y| x - y), Op::Sub(a, b)),
                BinOpKind::Mul => (ew(av, bv, |x, y| x * y), Op::Mul(a, b)),
                BinOpKind::Div => (ew(av, bv, |x, y| x / y), Op::Div(a, b)),
                BinOpKind::Pow => (ew(av, bv, |x, y| x.powf(y)), Op::Pow(a, b)),
                BinOpKind::MatMul => (matmul_forward(av, bv)?, Op::MatMul(a, b)),
                // 比較演算は微分不可。grad のスコープに現れたらエラー。
                _ => {
                    return Err(EvalError::InvalidArgument(
                        "比較演算子は微分できません".to_string(),
                    ))
                }
            }
        };
        let id = t.nodes.len();
        t.nodes.push(Node {
            value: val,
            op: node_op,
        });
        Ok(id)
    })
}

// 単項演算ノードを積む共通処理。
fn unary(a: usize, compute: impl Fn(&ArrayD<f64>) -> ArrayD<f64>, op_of: impl Fn(usize) -> Op) -> usize {
    with_top(|t| {
        let val = compute(&t.nodes[a].value);
        let id = t.nodes.len();
        t.nodes.push(Node {
            value: val,
            op: op_of(a),
        });
        id
    })
}

pub fn neg(a: usize) -> usize {
    unary(a, |v| v.mapv(|x| -x), Op::Neg)
}
pub fn exp(a: usize) -> usize {
    unary(a, |v| v.mapv(f64::exp), Op::Exp)
}
pub fn log(a: usize) -> usize {
    unary(a, |v| v.mapv(f64::ln), Op::Log)
}
pub fn tanh(a: usize) -> usize {
    unary(a, |v| v.mapv(f64::tanh), Op::Tanh)
}
pub fn sqrt(a: usize) -> usize {
    unary(a, |v| v.mapv(f64::sqrt), Op::Sqrt)
}

// 総和（スカラーへ縮約）ノードを積む。
pub fn sum(a: usize) -> usize {
    unary(a, |v| scalar(v.sum()), Op::Sum)
}

// 平均（スカラーへ縮約）ノードを積む。
pub fn mean(a: usize) -> usize {
    unary(
        a,
        |v| {
            let n = v.len().max(1) as f64;
            scalar(v.sum() / n)
        },
        Op::Mean,
    )
}

// 出力ノードから逆伝播して、全ノードの勾配を返す。
// grad は grads[入力ノード index] を取り出して使う。
// 出力はスカラー（mean/sum）を想定し、その随伴を ones（=1.0）で初期化する。
pub fn backward(output: usize) -> Vec<ArrayD<f64>> {
    with_top(|t| backward_impl(t, output))
}

fn backward_impl(tape: &Tape, output: usize) -> Vec<ArrayD<f64>> {
    let n = tape.nodes.len();
    // 各ノードの随伴を 0 で初期化（shape は forward 値に一致）。
    let mut grads: Vec<ArrayD<f64>> = tape
        .nodes
        .iter()
        .map(|nd| ArrayD::zeros(nd.value.raw_dim()))
        .collect();
    // 出力ノードの随伴を 1 で種付け。
    grads[output] = ArrayD::ones(tape.nodes[output].value.raw_dim());

    // ノードは生成順（親→子）に積まれているので、逆順に辿れば子→親の順になる。
    for i in (0..n).rev() {
        let g = grads[i].clone();
        let nodes = &tape.nodes;
        match nodes[i].op {
            Op::Leaf => {}
            Op::Add(a, b) => {
                acc(&mut grads, a, reduce_to(&g, &nodes[a].value));
                acc(&mut grads, b, reduce_to(&g, &nodes[b].value));
            }
            Op::Sub(a, b) => {
                acc(&mut grads, a, reduce_to(&g, &nodes[a].value));
                acc(&mut grads, b, reduce_to(&g.mapv(|v| -v), &nodes[b].value));
            }
            Op::Mul(a, b) => {
                let ga = ew(&g, &nodes[b].value, |x, y| x * y);
                let gb = ew(&g, &nodes[a].value, |x, y| x * y);
                acc(&mut grads, a, reduce_to(&ga, &nodes[a].value));
                acc(&mut grads, b, reduce_to(&gb, &nodes[b].value));
            }
            Op::Div(a, b) => {
                let av = &nodes[a].value;
                let bv = &nodes[b].value;
                // d/da = g / b
                let ga = ew(&g, bv, |x, y| x / y);
                // d/db = g * (-a / b^2)
                let local = ew(av, bv, |x, y| -x / (y * y));
                let gb = ew(&g, &local, |x, y| x * y);
                acc(&mut grads, a, reduce_to(&ga, av));
                acc(&mut grads, b, reduce_to(&gb, bv));
            }
            Op::Pow(a, b) => {
                let av = &nodes[a].value;
                let bv = &nodes[b].value;
                let cv = &nodes[i].value;
                // d/da = g * b * a^(b-1)
                let local_a = ew(av, bv, |x, y| y * x.powf(y - 1.0));
                let da = ew(&g, &local_a, |x, y| x * y);
                // d/db = g * c * ln(a)。指数が定数の場合この勾配は読まれない。
                let local_b = ew(cv, av, |c, a| if a > 0.0 { c * a.ln() } else { 0.0 });
                let db = ew(&g, &local_b, |x, y| x * y);
                acc(&mut grads, a, reduce_to(&da, av));
                acc(&mut grads, b, reduce_to(&db, bv));
            }
            Op::MatMul(a, b) => {
                matmul_backward(&mut grads, &g, a, b, nodes);
            }
            Op::Neg(a) => {
                acc(&mut grads, a, g.mapv(|v| -v));
            }
            Op::Exp(a) => {
                // d/da = g * exp(a) = g * c
                let cv = &nodes[i].value;
                acc(&mut grads, a, ew(&g, cv, |x, y| x * y));
            }
            Op::Log(a) => {
                let av = &nodes[a].value;
                acc(&mut grads, a, ew(&g, av, |x, y| x / y));
            }
            Op::Tanh(a) => {
                // d/da = g * (1 - tanh(a)^2) = g * (1 - c^2)
                let cv = &nodes[i].value;
                let local = cv.mapv(|c| 1.0 - c * c);
                acc(&mut grads, a, ew(&g, &local, |x, y| x * y));
            }
            Op::Sqrt(a) => {
                // d/da = g * 0.5 / sqrt(a) = g * 0.5 / c
                let cv = &nodes[i].value;
                let local = cv.mapv(|c| 0.5 / c);
                acc(&mut grads, a, ew(&g, &local, |x, y| x * y));
            }
            Op::Sum(a) => {
                // 出力スカラーの随伴を全要素へ等しく配る。
                let s = g.first().copied().unwrap_or(0.0);
                acc(&mut grads, a, ArrayD::from_elem(nodes[a].value.raw_dim(), s));
            }
            Op::Mean(a) => {
                let av = &nodes[a].value;
                let cnt = av.len().max(1) as f64;
                let s = g.first().copied().unwrap_or(0.0) / cnt;
                acc(&mut grads, a, ArrayD::from_elem(av.raw_dim(), s));
            }
        }
    }
    grads
}

// 随伴を累積する（同じノードへ複数経路から勾配が流れ込む場合に加算）。
// contribution の shape は grads[idx]（= ノードの forward 値）に一致している前提。
fn acc(grads: &mut [ArrayD<f64>], idx: usize, contribution: ArrayD<f64>) {
    grads[idx] = &grads[idx] + &contribution;
}

// 要素ごと二項演算。scalar（0 次元）↔ tensor のブロードキャストに対応する。
// どちらも tensor の場合は同一 shape を前提に zip する。
fn ew(a: &ArrayD<f64>, b: &ArrayD<f64>, f: impl Fn(f64, f64) -> f64) -> ArrayD<f64> {
    if a.ndim() == 0 {
        let s = a.first().copied().unwrap_or(0.0);
        b.mapv(|x| f(s, x))
    } else if b.ndim() == 0 {
        let s = b.first().copied().unwrap_or(0.0);
        a.mapv(|x| f(x, s))
    } else {
        ndarray::Zip::from(a).and(b).map_collect(|&x, &y| f(x, y))
    }
}

// 上流の勾配 g を target の shape に縮約する。
// ブロードキャストの逆操作: target がスカラーなら g を総和、g がスカラーなら
// target の shape に複製する。同一 shape ならそのまま。
fn reduce_to(g: &ArrayD<f64>, target: &ArrayD<f64>) -> ArrayD<f64> {
    if g.raw_dim() == target.raw_dim() {
        g.clone()
    } else if target.ndim() == 0 {
        scalar(g.sum())
    } else if g.ndim() == 0 {
        let s = g.first().copied().unwrap_or(0.0);
        ArrayD::from_elem(target.raw_dim(), s)
    } else {
        // 想定外の shape 不一致。安全側で複製を返す（通常到達しない）。
        g.clone()
    }
}

// 行列積の forward。対応ケース: 2D×2D → 2D、2D×1D → 1D。
fn matmul_forward(a: &ArrayD<f64>, b: &ArrayD<f64>) -> Result<ArrayD<f64>, EvalError> {
    match (a.ndim(), b.ndim()) {
        (2, 2) => {
            let a2 = a.view().into_dimensionality::<Ix2>().unwrap();
            let b2 = b.view().into_dimensionality::<Ix2>().unwrap();
            if a2.shape()[1] != b2.shape()[0] {
                return Err(EvalError::TensorShapeMismatch {
                    op: "@",
                    a: a2.shape().to_vec(),
                    b: b2.shape().to_vec(),
                });
            }
            Ok(a2.dot(&b2).into_dyn())
        }
        (2, 1) => {
            let a2 = a.view().into_dimensionality::<Ix2>().unwrap();
            let b1 = b.view().into_dimensionality::<Ix1>().unwrap();
            if a2.shape()[1] != b1.len() {
                return Err(EvalError::TensorShapeMismatch {
                    op: "@",
                    a: a2.shape().to_vec(),
                    b: b1.shape().to_vec(),
                });
            }
            Ok(a2.dot(&b1).into_dyn())
        }
        (da, db) => Err(EvalError::InvalidArgument(format!(
            "@ は 2D×2D または 2D×1D のみ対応（{da}D × {db}D は未対応）"
        ))),
    }
}

// 行列積の backward。C = A @ B に対し A, B の随伴を計算する。
//   2D×2D: dA = g @ B^T, dB = A^T @ g
//   2D×1D: dA = outer(g, B), dB = A^T @ g
fn matmul_backward(grads: &mut [ArrayD<f64>], g: &ArrayD<f64>, a: usize, b: usize, nodes: &[Node]) {
    let av = &nodes[a].value;
    let bv = &nodes[b].value;
    match (av.ndim(), bv.ndim()) {
        (2, 2) => {
            let a2 = av.view().into_dimensionality::<Ix2>().unwrap();
            let b2 = bv.view().into_dimensionality::<Ix2>().unwrap();
            let g2 = g.view().into_dimensionality::<Ix2>().unwrap();
            let da = g2.dot(&b2.t());
            let db = a2.t().dot(&g2);
            acc(grads, a, da.into_dyn());
            acc(grads, b, db.into_dyn());
        }
        (2, 1) => {
            let a2 = av.view().into_dimensionality::<Ix2>().unwrap();
            let b1 = bv.view().into_dimensionality::<Ix1>().unwrap();
            let g1 = g.view().into_dimensionality::<Ix1>().unwrap();
            let (m, k) = (a2.shape()[0], a2.shape()[1]);
            // dA = outer(g, B): dA[i,j] = g[i] * B[j]
            let mut da = ndarray::Array2::<f64>::zeros((m, k));
            for i in 0..m {
                for j in 0..k {
                    da[[i, j]] = g1[i] * b1[j];
                }
            }
            // dB = A^T @ g
            let db = a2.t().dot(&g1);
            acc(grads, a, da.into_dyn());
            acc(grads, b, db.into_dyn());
        }
        _ => {
            // forward が拒否する shape なので backward では到達しない。
        }
    }
}
