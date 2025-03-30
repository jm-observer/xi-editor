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
use lapce_xi_rope::Rope;

fn main() {
    let mut a = Rope::from("hello.");
    a.edit(5..6, "!");
    a.print_tree(0);
    for i in 0..1000 {
        let l = a.len();
        a.edit(l..l, &(i.to_string() + "\n"));
    }
    a.print_tree(0);
    let l = a.len();
    for s in a.clone().iter_chunks(100..300) {
        println!("\nchunk {:?}", s);
    }
    a.edit(1000..l, "");
    //a = a.subrange(0, 1000);
    println!("\n{:?}", String::from(a));
}
