// Copyright 2016 The xi-editor Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! A general b-tree structure suitable for ropes and the like.

use std::cmp::{min, Ordering};
use std::sync::Arc;

use crate::interval::{Interval, IntervalBounds};

const MIN_CHILDREN: usize = 4;
const MAX_CHILDREN: usize = 8;

pub trait NodeInfo: Clone {
    /// The type of the leaf.
    ///
    /// A given `NodeInfo` is for exactly one type of leaf. That is why
    /// the leaf type is an associated type rather than a type parameter.
    type L: Leaf;

    /// An operator that combines info from two subtrees. It is intended
    /// (but not strictly enforced) that this operator be associative and
    /// obey an identity property. In mathematical terms, the accumulate
    /// method is the operation of a monoid.
    fn accumulate(&mut self, other: &Self);

    /// A mapping from a leaf into the info type. It is intended (but
    /// not strictly enforced) that applying the accumulate method to
    /// the info derived from two leaves gives the same result as
    /// deriving the info from the concatenation of the two leaves. In
    /// mathematical terms, the compute_info method is a monoid
    /// homomorphism.
    fn compute_info(_: &Self::L) -> Self;

    /// The identity of the monoid. Need not be implemented because it
    /// can be computed from the leaf default.
    ///
    /// This is here to demonstrate that this is a monoid.
    fn identity() -> Self {
        Self::compute_info(&Self::L::default())
    }

    /// The interval covered by the first `len` base units of this node. The
    /// default impl is sufficient for most types, but interval trees may need
    /// to override it.
    fn interval(&self, len: usize) -> Interval {
        Interval::new(0, len)
    }
}

/// A trait indicating the default metric of a NodeInfo.
///
/// Adds quality of life functions to
/// Node\<N\>, where N is a DefaultMetric.
/// For example, [Node\<DefaultMetric\>.count](struct.Node.html#method.count).
pub trait DefaultMetric: NodeInfo {
    type DefaultMetric: Metric<Self>;
}

/// A trait for the leaves of trees of type [Node](struct.Node.html).
///
/// Two leafs can be concatenated using `push_maybe_split`.
pub trait Leaf: Sized + Clone + Default {
    /// Measurement of leaf in base units.
    /// A 'base unit' refers to the smallest discrete unit
    /// by which a given concrete type can be indexed.
    /// Concretely, for Rust's String type the base unit is the byte.
    fn len(&self) -> usize;

    /// Generally a minimum size requirement for leaves.
    fn is_ok_child(&self) -> bool;

    /// Combine the part `other` denoted by the `Interval` `iv` into `self`,
    /// optionly splitting off a new `Leaf` if `self` would have become too big.
    /// Returns either `None` if no splitting was needed, or `Some(rest)` if
    /// `rest` was split off.
    ///
    /// Interval is in "base units".  Generally implements a maximum size.
    ///
    /// # Invariants:
    /// - If one or the other input is empty, then no split.
    /// - If either input satisfies `is_ok_child`, then, on return, `self`
    ///   satisfies this, as does the optional split.
    fn push_maybe_split(&mut self, other: &Self, iv: Interval) -> Option<Self>;

    /// Same meaning as push_maybe_split starting from an empty
    /// leaf, but maybe can be implemented more efficiently?
    ///
    // TODO: remove if it doesn't pull its weight
    fn subseq(&self, iv: Interval) -> Self {
        let mut result = Self::default();
        if result.push_maybe_split(self, iv).is_some() {
            panic!("unexpected split");
        }
        result
    }
}

/// A b-tree node storing leaves at the bottom, and with info
/// retained at each node. It is implemented with atomic reference counting
/// and copy-on-write semantics, so an immutable clone is a very cheap
/// operation, and nodes can be shared across threads. Even so, it is
/// designed to be updated in place, with efficiency similar to a mutable
/// data structure, using uniqueness of reference count to detect when
/// this operation is safe.
///
/// When the leaf is a string, this is a rope data structure (a persistent
/// rope in functional programming jargon). However, it is not restricted
/// to strings, and it is expected to be the basis for a number of data
/// structures useful for text processing.
#[derive(Clone)]
pub struct Node<N: NodeInfo>(Arc<NodeBody<N>>);

#[derive(Clone)]
struct NodeBody<N: NodeInfo> {
    height: usize,
    len: usize,
    info: N,
    val: NodeVal<N>,
}

#[derive(Clone)]
enum NodeVal<N: NodeInfo> {
    Leaf(N::L),
    Internal(Vec<Node<N>>),
}

// also consider making Metric a newtype for usize, so type system can
// help separate metrics

/// A trait for quickly processing attributes of a
/// [NodeInfo](struct.NodeInfo.html).
///
/// For the conceptual background see the
/// [blog post, Rope science, part 2: metrics](https://github.com/google/xi-editor/blob/master/docs/docs/rope_science_02.md).
pub trait Metric<N: NodeInfo> {
    /// Return the size of the
    /// [NodeInfo::L](trait.NodeInfo.html#associatedtype.L), as measured by this
    /// metric.
    ///
    /// The usize argument is the total size/length of the node, in base units.
    ///
    /// # Examples
    /// For the [LinesMetric](../rope/struct.LinesMetric.html), this gives the number of
    /// lines in string contained in the leaf. For the
    /// [BaseMetric](../rope/struct.BaseMetric.html), this gives the size of the string
    /// in uft8 code units, that is, bytes.
    ///
    fn measure(info: &N, len: usize) -> usize;

    /// Returns the smallest offset, in base units, for an offset in measured units.
    ///
    /// # Invariants:
    ///
    /// - `from_base_units(to_base_units(x)) == x` is True for valid `x`
    fn to_base_units(l: &N::L, in_measured_units: usize) -> usize;

    /// Returns the smallest offset in measured units corresponding to an offset in base units.
    ///
    /// # Invariants:
    ///
    /// - `from_base_units(to_base_units(x)) == x` is True for valid `x`
    fn from_base_units(l: &N::L, in_base_units: usize) -> usize;

    /// Return whether the offset in base units is a boundary of this metric.
    /// If a boundary is at end of a leaf then this method must return true.
    /// However, a boundary at the beginning of a leaf is optional
    /// (the previous leaf will be queried).
    fn is_boundary(l: &N::L, offset: usize) -> bool;

    /// Returns the index of the boundary directly preceding offset,
    /// or None if no such boundary exists. Input and result are in base units.
    fn prev(l: &N::L, offset: usize) -> Option<usize>;

    /// Returns the index of the first boundary for which index > offset,
    /// or None if no such boundary exists. Input and result are in base units.
    fn next(l: &N::L, offset: usize) -> Option<usize>;

    /// Returns true if the measured units in this metric can span multiple
    /// leaves.  As an example, in a metric that measures lines in a rope, a
    /// line may start in one leaf and end in another; however in a metric
    /// measuring bytes, storage of a single byte cannot extend across leaves.
    fn can_fragment() -> bool;
}

impl<N: NodeInfo> Node<N> {
    pub fn from_leaf(l: N::L) -> Node<N> {
        let len = l.len();
        let info = N::compute_info(&l);
        Node(Arc::new(NodeBody {
            height: 0,
            len,
            info,
            val: NodeVal::Leaf(l),
        }))
    }

    /// Create a node from a vec of nodes.
    ///
    /// The input must satisfy the following balancing requirements:
    /// * The length of `nodes` must be <= MAX_CHILDREN and > 1.
    /// * All the nodes are the same height.
    /// * All the nodes must satisfy is_ok_child.
    fn from_nodes(nodes: Vec<Node<N>>) -> Node<N> {
        debug_assert!(nodes.len() > 1);
        debug_assert!(nodes.len() <= MAX_CHILDREN);
        let height = nodes[0].0.height + 1;
        let mut len = nodes[0].0.len;
        let mut info = nodes[0].0.info.clone();
        debug_assert!(nodes[0].is_ok_child());
        for child in &nodes[1..] {
            debug_assert_eq!(child.height() + 1, height);
            debug_assert!(child.is_ok_child());
            len += child.0.len;
            info.accumulate(&child.0.info);
        }
        Node(Arc::new(NodeBody {
            height,
            len,
            info,
            val: NodeVal::Internal(nodes),
        }))
    }

    pub fn len(&self) -> usize {
        self.0.len
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns `true` if these two `Node`s share the same underlying data.
    ///
    /// This is principally intended to be used by the druid crate, without needing
    /// to actually add a feature and implement druid's `Data` trait.
    pub fn ptr_eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }

    pub(crate) fn height(&self) -> usize {
        self.0.height
    }

    fn is_leaf(&self) -> bool {
        self.0.height == 0
    }

    fn interval(&self) -> Interval {
        self.0.info.interval(self.0.len)
    }

    pub(crate) fn get_children(&self) -> &[Node<N>] {
        if let NodeVal::Internal(ref v) = self.0.val {
            v
        } else {
            panic!("get_children called on leaf node");
        }
    }

    pub(crate) fn get_leaf(&self) -> &N::L {
        if let NodeVal::Leaf(ref l) = self.0.val {
            l
        } else {
            panic!("get_leaf called on internal node");
        }
    }

    /// Call a callback with a mutable reference to a leaf.
    ///
    /// This clones the leaf if the reference is shared. It also recomputes
    /// length and info after the leaf is mutated.
    fn with_leaf_mut<T>(&mut self, f: impl FnOnce(&mut N::L) -> T) -> T {
        let inner = Arc::make_mut(&mut self.0);
        if let NodeVal::Leaf(ref mut l) = inner.val {
            let result = f(l);
            inner.len = l.len();
            inner.info = N::compute_info(l);
            result
        } else {
            panic!("with_leaf_mut called on internal node");
        }
    }

    fn is_ok_child(&self) -> bool {
        match self.0.val {
            NodeVal::Leaf(ref l) => l.is_ok_child(),
            NodeVal::Internal(ref nodes) => nodes.len() >= MIN_CHILDREN,
        }
    }

    fn merge_nodes(children1: &[Node<N>], children2: &[Node<N>]) -> Node<N> {
        let n_children = children1.len() + children2.len();
        if n_children <= MAX_CHILDREN {
            Node::from_nodes([children1, children2].concat())
        } else {
            // Note: this leans left. Splitting at midpoint is also an option
            let splitpoint = min(MAX_CHILDREN, n_children - MIN_CHILDREN);
            let mut iter = children1.iter().chain(children2.iter()).cloned();
            let left = iter.by_ref().take(splitpoint).collect();
            let right = iter.collect();
            let parent_nodes = vec![Node::from_nodes(left), Node::from_nodes(right)];
            Node::from_nodes(parent_nodes)
        }
    }

    fn merge_leaves(mut rope1: Node<N>, rope2: Node<N>) -> Node<N> {
        debug_assert!(rope1.is_leaf() && rope2.is_leaf());

        let both_ok = rope1.get_leaf().is_ok_child() && rope2.get_leaf().is_ok_child();
        if both_ok {
            return Node::from_nodes(vec![rope1, rope2]);
        }
        let res = {
            let node1 = Arc::make_mut(&mut rope1.0);
            let leaf2 = rope2.get_leaf();
            if let NodeVal::Leaf(ref mut leaf1) = node1.val {
                let leaf2_iv = Interval::new(0, leaf2.len());
                let new = leaf1.push_maybe_split(leaf2, leaf2_iv);
                node1.len = leaf1.len();
                node1.info = N::compute_info(leaf1);
                new
            } else {
                panic!("merge_leaves called on non-leaf");
            }
        };
        match res {
            Some(new) => Node::from_nodes(vec![rope1, Node::from_leaf(new)]),
            None => rope1,
        }
    }

    pub fn concat(rope1: Node<N>, rope2: Node<N>) -> Node<N> {
        let h1 = rope1.height();
        let h2 = rope2.height();

        match h1.cmp(&h2) {
            Ordering::Less => {
                let children2 = rope2.get_children();
                if h1 == h2 - 1 && rope1.is_ok_child() {
                    return Node::merge_nodes(&[rope1], children2);
                }
                let newrope = Node::concat(rope1, children2[0].clone());
                if newrope.height() == h2 - 1 {
                    Node::merge_nodes(&[newrope], &children2[1..])
                } else {
                    Node::merge_nodes(newrope.get_children(), &children2[1..])
                }
            }
            Ordering::Equal => {
                if rope1.is_ok_child() && rope2.is_ok_child() {
                    return Node::from_nodes(vec![rope1, rope2]);
                }
                if h1 == 0 {
                    return Node::merge_leaves(rope1, rope2);
                }
                Node::merge_nodes(rope1.get_children(), rope2.get_children())
            }
            Ordering::Greater => {
                let children1 = rope1.get_children();
                if h2 == h1 - 1 && rope2.is_ok_child() {
                    return Node::merge_nodes(children1, &[rope2]);
                }
                let lastix = children1.len() - 1;
                let newrope = Node::concat(children1[lastix].clone(), rope2);
                if newrope.height() == h1 - 1 {
                    Node::merge_nodes(&children1[..lastix], &[newrope])
                } else {
                    Node::merge_nodes(&children1[..lastix], newrope.get_children())
                }
            }
        }
    }

    pub fn measure<M: Metric<N>>(&self) -> usize {
        M::measure(&self.0.info, self.0.len)
    }

    pub(crate) fn push_subseq(&self, b: &mut TreeBuilder<N>, iv: Interval) {
        if iv.is_empty() {
            return;
        }
        if iv == self.interval() {
            b.push(self.clone());
            return;
        }
        match self.0.val {
            NodeVal::Leaf(ref l) => {
                b.push_leaf_slice(l, iv);
            }
            NodeVal::Internal(ref v) => {
                let mut offset = 0;
                for child in v {
                    if iv.is_before(offset) {
                        break;
                    }
                    let child_iv = child.interval();
                    // easier just to use signed ints?
                    let rec_iv = iv
                        .intersect(child_iv.translate(offset))
                        .translate_neg(offset);
                    child.push_subseq(b, rec_iv);
                    offset += child.len();
                }
            }
        }
    }

    pub fn subseq<T: IntervalBounds>(&self, iv: T) -> Node<N> {
        let iv = iv.into_interval(self.len());
        let mut b = TreeBuilder::new();
        self.push_subseq(&mut b, iv);
        b.build()
    }

    pub fn edit<T, IV>(&mut self, iv: IV, new: T)
    where
        T: Into<Node<N>>,
        IV: IntervalBounds,
    {
        let mut b = TreeBuilder::new();
        let iv = iv.into_interval(self.len());
        let self_iv = self.interval();
        self.push_subseq(&mut b, self_iv.prefix(iv));
        b.push(new.into());
        self.push_subseq(&mut b, self_iv.suffix(iv));
        *self = b.build();
    }

    // doesn't deal with endpoint, handle that specially if you need it
    pub fn convert_metrics<M1: Metric<N>, M2: Metric<N>>(&self, mut m1: usize) -> usize {
        if m1 == 0 {
            return 0;
        }
        // If M1 can fragment, then we must land on the leaf containing
        // the m1 boundary. Otherwise, we can land on the beginning of
        // the leaf immediately following the M1 boundary, which may be
        // more efficient.
        let m1_fudge = if M1::can_fragment() { 1 } else { 0 };
        let mut m2 = 0;
        let mut node = self;
        while node.height() > 0 {
            for child in node.get_children() {
                let child_m1 = child.measure::<M1>();
                if m1 < child_m1 + m1_fudge {
                    node = child;
                    break;
                }
                m2 += child.measure::<M2>();
                m1 -= child_m1;
            }
        }
        let l = node.get_leaf();
        let base = M1::to_base_units(l, m1);
        m2 + M2::from_base_units(l, base)
    }
}

use std::fmt::Debug;

impl<N: NodeInfo> Node<N>
where
    N: Debug,
    N::L: Debug,
{
    pub fn print_tree(&self, indent: usize) {
        let node = &*self.0; // 解引用 Arc<NodeBody<N>>

        let indent_str = "  ".repeat(indent);
        let node_type = match &node.val {
            NodeVal::Leaf(_) => "Leaf",
            NodeVal::Internal(_) => "Internal",
        };

        println!(
            "{}[H={}] {} len={} info={:?}",
            indent_str, node.height, node_type, node.len, node.info
        );

        match &node.val {
            NodeVal::Internal(children) => {
                for child in children {
                    child.print_tree(indent + 1);
                }
            }
            NodeVal::Leaf(leaf) => {
                println!("{}  └─ Leaf content: {:?}", indent_str, leaf);
            }
        }
    }
}

impl<N: DefaultMetric> Node<N> {
    /// Measures the length of the text bounded by ``DefaultMetric::measure(offset)`` with another metric.
    ///
    /// # Examples
    /// ```
    /// use crate::xi_rope::{Rope, LinesMetric};
    ///
    /// // the default metric of Rope is BaseMetric (aka number of bytes)
    /// let my_rope = Rope::from("first line \n second line \n");
    ///
    /// // count the number of lines in my_rope
    /// let num_lines = my_rope.count::<LinesMetric>(my_rope.len());
    /// assert_eq!(2, num_lines);
    /// ```
    pub fn count<M: Metric<N>>(&self, offset: usize) -> usize {
        self.convert_metrics::<N::DefaultMetric, M>(offset)
    }

    /// Measures the length of the text bounded by ``M::measure(offset)`` with the default metric.
    ///
    /// # Examples
    /// ```
    /// use crate::xi_rope::{Rope, LinesMetric};
    ///
    /// // the default metric of Rope is BaseMetric (aka number of bytes)
    /// let my_rope = Rope::from("first line \n second line \n");
    ///
    /// // get the byte offset of the line at index 1
    /// let byte_offset = my_rope.count_base_units::<LinesMetric>(1);
    /// assert_eq!(12, byte_offset);
    /// ```
    pub fn count_base_units<M: Metric<N>>(&self, offset: usize) -> usize {
        self.convert_metrics::<M, N::DefaultMetric>(offset)
    }
}

impl<N: NodeInfo> Default for Node<N> {
    fn default() -> Node<N> {
        Node::from_leaf(N::L::default())
    }
}

/// A builder for creating new trees.
pub struct TreeBuilder<N: NodeInfo> {
    // A stack of partially built trees. These are kept in order of
    // strictly descending height, and all vectors have a length less
    // than MAX_CHILDREN and greater than zero.
    //
    // In addition, there is a balancing invariant: for each vector
    // of length greater than one, all elements satisfy `is_ok_child`.
    stack: Vec<Vec<Node<N>>>,
}

impl<N: NodeInfo> TreeBuilder<N> {
    /// A new, empty builder.
    pub fn new() -> TreeBuilder<N> {
        TreeBuilder { stack: Vec::new() }
    }

    /// Append a node to the tree being built.
    pub fn push(&mut self, mut n: Node<N>) {
        loop {
            let ord = if let Some(last) = self.stack.last() {
                last[0].height().cmp(&n.height())
            } else {
                Ordering::Greater
            };
            match ord {
                Ordering::Less => {
                    n = Node::concat(self.pop(), n);
                }
                Ordering::Equal => {
                    let tos = self.stack.last_mut().unwrap();
                    if tos.last().unwrap().is_ok_child() && n.is_ok_child() {
                        tos.push(n);
                    } else if n.height() == 0 {
                        let iv = Interval::new(0, n.len());
                        let new_leaf = tos
                            .last_mut()
                            .unwrap()
                            .with_leaf_mut(|l| l.push_maybe_split(n.get_leaf(), iv));
                        if let Some(new_leaf) = new_leaf {
                            tos.push(Node::from_leaf(new_leaf));
                        }
                    } else {
                        let last = tos.pop().unwrap();
                        let children1 = last.get_children();
                        let children2 = n.get_children();
                        let n_children = children1.len() + children2.len();
                        if n_children <= MAX_CHILDREN {
                            tos.push(Node::from_nodes([children1, children2].concat()));
                        } else {
                            // Note: this leans left. Splitting at midpoint is also an option
                            let splitpoint = min(MAX_CHILDREN, n_children - MIN_CHILDREN);
                            let mut iter = children1.iter().chain(children2.iter()).cloned();
                            let left = iter.by_ref().take(splitpoint).collect();
                            let right = iter.collect();
                            tos.push(Node::from_nodes(left));
                            tos.push(Node::from_nodes(right));
                        }
                    }
                    if tos.len() < MAX_CHILDREN {
                        break;
                    }
                    n = self.pop()
                }
                Ordering::Greater => {
                    self.stack.push(vec![n]);
                    break;
                }
            }
        }
    }

    /// Append a sequence of leaves.
    pub fn push_leaves(&mut self, leaves: impl IntoIterator<Item = N::L>) {
        for leaf in leaves.into_iter() {
            self.push(Node::from_leaf(leaf));
        }
    }

    /// Append a single leaf.
    pub fn push_leaf(&mut self, l: N::L) {
        self.push(Node::from_leaf(l))
    }

    /// Append a slice of a single leaf.
    pub fn push_leaf_slice(&mut self, l: &N::L, iv: Interval) {
        self.push(Node::from_leaf(l.subseq(iv)))
    }

    /// Build the final tree.
    ///
    /// The tree is the concatenation of all the nodes and leaves that have been pushed
    /// on the builder, in order.
    pub fn build(mut self) -> Node<N> {
        if self.stack.is_empty() {
            Node::from_leaf(N::L::default())
        } else {
            let mut n = self.pop();
            while !self.stack.is_empty() {
                n = Node::concat(self.pop(), n);
            }
            n
        }
    }

    /// Pop the last vec-of-nodes off the stack, resulting in a node.
    fn pop(&mut self) -> Node<N> {
        let nodes = self.stack.pop().unwrap();
        if nodes.len() == 1 {
            nodes.into_iter().next().unwrap()
        } else {
            Node::from_nodes(nodes)
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{rope::*, Cursor};

    fn build_triangle(n: u32) -> String {
        let mut s = String::new();
        let mut line = String::new();
        for _ in 0..n {
            s += &line;
            s += "\n";
            line += "a";
        }
        s
    }

    #[test]
    fn eq_rope_with_pieces() {
        let n = 2_000;
        let s = build_triangle(n);
        let mut builder_default = TreeBuilder::new();
        let mut concat_rope = Rope::default();
        builder_default.push_str(&s);
        let mut i = 0;
        while i < s.len() {
            let j = (i + 1000).min(s.len());
            concat_rope = concat_rope + s[i..j].into();
            i = j;
        }
        let built_rope = builder_default.build();
        assert_eq!(built_rope, concat_rope);
    }

    #[test]
    fn cursor_next_triangle() {
        let n = 2_000;
        let text = Rope::from(build_triangle(n));

        let mut cursor = Cursor::new_unsafe(&text, 0);
        let mut prev_offset = cursor.pos();
        for i in 1..(n + 1) as usize {
            let offset = cursor
                .next::<LinesMetric>()
                .expect("arrived at the end too soon");
            assert_eq!(offset - prev_offset, i);
            prev_offset = offset;
        }
        assert_eq!(cursor.next::<LinesMetric>(), None);
    }

    #[test]
    fn node_is_empty() {
        let text = Rope::from(String::new());
        assert_eq!(text.is_empty(), true);
    }

    #[test]
    fn cursor_next_empty() {
        let text = Rope::from(String::new());
        let mut cursor = Cursor::new_unsafe(&text, 0);
        assert_eq!(cursor.next::<LinesMetric>(), None);
        assert_eq!(cursor.pos(), 0);
    }

    #[test]
    fn cursor_iter() {
        let text: Rope = build_triangle(50).into();
        let mut cursor = Cursor::new_unsafe(&text, 0);
        let mut manual = Vec::new();
        while let Some(nxt) = cursor.next::<LinesMetric>() {
            manual.push(nxt);
        }

        cursor.set_unsafe(0);
        let auto = cursor.iter::<LinesMetric>().collect::<Vec<_>>();
        assert_eq!(manual, auto);
    }

    #[test]
    fn cursor_next_misc() {
        cursor_next_for("toto");
        cursor_next_for("toto\n");
        cursor_next_for("toto\ntata");
        cursor_next_for("歴史\n科学的");
        cursor_next_for("\n歴史\n科学的\n");
        cursor_next_for(&build_triangle(100));
    }

    fn cursor_next_for(s: &str) {
        let r = Rope::from(s.to_owned());
        for i in 0..r.len() {
            let mut c = Cursor::new_unsafe(&r, i);
            let it = c.next::<LinesMetric>();
            let pos = c.pos();
            assert!(
                s.as_bytes()[i..pos - 1].iter().all(|c| *c != b'\n'),
                "missed linebreak"
            );
            if pos < s.len() {
                assert!(it.is_some(), "must be Some(_)");
                assert!(s.as_bytes()[pos - 1] == b'\n', "not a linebreak");
            } else if s.as_bytes()[s.len() - 1] == b'\n' {
                assert!(it.is_some(), "must be Some(_)");
            } else {
                assert!(it.is_none());
                assert!(c.get_leaf().is_none());
            }
        }
    }

    #[test]
    fn cursor_prev_misc() {
        cursor_prev_for("toto");
        cursor_prev_for("a\na\n");
        cursor_prev_for("toto\n");
        cursor_prev_for("toto\ntata");
        cursor_prev_for("歴史\n科学的");
        cursor_prev_for("\n歴史\n科学的\n");
        cursor_prev_for(&build_triangle(100));
    }

    fn cursor_prev_for(s: &str) {
        let r = Rope::from(s.to_owned());
        for i in 0..r.len() {
            let mut c = Cursor::new_unsafe(&r, i);
            let it = c.prev::<LinesMetric>();
            let pos = c.pos();

            //Should countain at most one linebreak
            assert!(
                s.as_bytes()[pos..i].iter().filter(|c| **c == b'\n').count() <= 1,
                "missed linebreak"
            );

            if i == 0 && s.as_bytes()[i] == b'\n' {
                assert_eq!(pos, 0);
            }

            if pos > 0 {
                assert!(it.is_some(), "must be Some(_)");
                assert!(s.as_bytes()[pos - 1] == b'\n', "not a linebreak");
            }
        }
    }

    #[test]
    fn at_or_next() {
        let text: Rope = "this\nis\nalil\nstring".into();
        let mut cursor = Cursor::new_unsafe(&text, 0);
        assert_eq!(cursor.at_or_next::<LinesMetric>(), Some(5));
        assert_eq!(cursor.at_or_next::<LinesMetric>(), Some(5));
        cursor.set_unsafe(1);
        assert_eq!(cursor.at_or_next::<LinesMetric>(), Some(5));
        assert_eq!(cursor.at_or_prev::<LinesMetric>(), Some(5));
        cursor.set_unsafe(6);
        assert_eq!(cursor.at_or_prev::<LinesMetric>(), Some(5));
        cursor.set_unsafe(6);
        assert_eq!(cursor.at_or_next::<LinesMetric>(), Some(8));
        assert_eq!(cursor.at_or_next::<LinesMetric>(), Some(8));
    }

    #[test]
    fn next_zero_measure_large() {
        let mut text = Rope::from("a");
        for _ in 0..24 {
            text = Node::concat(text.clone(), text);
            let mut cursor = Cursor::new_unsafe(&text, 0);
            assert_eq!(cursor.next::<LinesMetric>(), None);
            // Test that cursor is properly invalidated and at end of text.
            assert_eq!(cursor.get_leaf(), None);
            assert_eq!(cursor.pos(), text.len());

            cursor.set_unsafe(text.len());
            assert_eq!(cursor.prev::<LinesMetric>(), None);
            // Test that cursor is properly invalidated and at beginning of text.
            assert_eq!(cursor.get_leaf(), None);
            assert_eq!(cursor.pos(), 0);
        }
    }

    #[test]
    fn prev_line_large() {
        let s: String = format!("{}{}", "\n", build_triangle(1000));
        let rope = Rope::from(s);
        let mut expected_pos = rope.len();
        let mut cursor = Cursor::new_unsafe(&rope, rope.len());

        for i in (1..1001).rev() {
            expected_pos = expected_pos - i;
            assert_eq!(expected_pos, cursor.prev::<LinesMetric>().unwrap());
        }

        assert_eq!(None, cursor.prev::<LinesMetric>());
    }

    #[test]
    fn prev_line_small() {
        let empty_rope = Rope::from("\n");
        let mut cursor = Cursor::new_unsafe(&empty_rope, empty_rope.len());
        assert_eq!(None, cursor.prev::<LinesMetric>());

        let rope = Rope::from("\n\n\n\n\n\n\n\n\n\n");
        cursor = Cursor::new_unsafe(&rope, rope.len());
        let mut expected_pos = rope.len();
        for _ in (1..10).rev() {
            expected_pos -= 1;
            assert_eq!(expected_pos, cursor.prev::<LinesMetric>().unwrap());
        }

        assert_eq!(None, cursor.prev::<LinesMetric>());
    }

    #[test]
    fn balance_invariant() {
        let mut tb = TreeBuilder::<RopeInfo>::new();
        let leaves: Vec<String> = (0..1000).map(|i| i.to_string().into()).collect();
        tb.push_leaves(leaves);
        let tree = tb.build();
        println!("height {}", tree.height());
    }
}
