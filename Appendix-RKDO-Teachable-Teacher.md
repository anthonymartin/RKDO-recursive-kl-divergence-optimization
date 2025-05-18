## Appendix: The Teachable Teacher - A Recursive Approach

![RKDO Visualization](./visualizations/data/rkdo_kaleidoscope_split_8.gif "Title")



At first glance, Recursive KL Divergence Optimization (RKDO) appears paradoxical. If the student $q$ is meant to align to a teacher $p$, why should the teacher shift? Shouldn't it hold still and define the target?

In RKDO, the role of the teacher is not fixed. Unlike traditional supervision, where the teacher provides a static target $p$ for the student $q$ to imitate, RKDO defines the teacher recursively:

$$
p^{(t)} = (1 - \alpha)p^{(t-1)} + \alpha q^{(t-1)}
$$

This update rule blends the teacher with the student over time. The teacher is not static—it **yields** and integrates prior student behavior into its own signal. This is not an oversight. It is the design.

The tradeoff is explicit: the teacher **sacrifices precision to preserve direction**. By adjusting its position, it avoids presenting the student with unreachable or unstable gradients. This maintains **gradient informativeness**, **directional coherence**, and under bounded assumptions, **provable convergence**. **In RKDO, the student learns more effectively because the teacher becomes teachable.**

---

### What the Visualization Shows

The RKDO Kaleidoscope animation renders this recursive learning process in motion. Each frame displays:

| Element             | Meaning                                                                 |                                                    |
| ------------------- | ----------------------------------------------------------------------- | -------------------------------------------------- |
| **Nodes**           | Data anchors with associated neighborhood conditionals                  |                                                    |
| **Edges**           | Directed transitions $i \to j$, weighted by student ( q(j\|i) ), colored by KL divergence $\text{KL}(p \| q)$ |                  |
| **Petal Triangles** | Local 3-class simplex showing the top-3 neighbor beliefs of $p$ and $q$ |                                                    |
| **Dots**            | Red = teacher $p$, Blue = student $q$                                   |                                                    |
| **Dashed Line**     | Direct Euclidean path from $q$ to $p$                                   |                                                    |
| **Purple Arrow**    | Euclidean gradient direction $p - q$                                    |                                                    |
| **Green Arrow**     | Natural gradient $F_q^{-1}(p - q)$, respecting information geometry     |                                                    |

These aren't decorative forms. Each arrow is a computed vector field in simplex space. Each edge's hue encodes divergence tension. Together, these elements show where the student and teacher disagree (edge color), how much the student believes in each transition (edge thickness), and how updates proceed (arrows within petals).

---

### Recursive Frame Alignment as Strategy

RKDO does not optimize in a static geometry. It **modulates** its supervisory frame as learning unfolds. The teacher is not a fixed oracle but a **drifting supervisory layer**—a smooth surface floating atop a slower-evolving representational substrate.

This is not artistic abstraction. RKDO's convergence proof shows that recursive updating leads to **faster, more stable alignment** than static targets. Empirically, it reaches lower loss values and requires fewer epochs than fixed-teacher approaches.

This structure allows:

* **Low-pass filtering** of student noise
* **Directional stability** across epochs
* **Preserved signal coherence** in early learning, where static targets fail

But this advantage has limits. Over time, if the teacher adapts too much, it becomes indistinguishable from the student. The gradient signal weakens. The system stops learning—not because it has succeeded, but because the tension that drives learning has disappeared.

To prevent this, RKDO can anneal $\alpha$ over time. In early training, a high $\alpha$ helps the teacher meet the student where it is. Later, a reduced $\alpha$ helps the teacher **hold its ground**, allowing the student to catch up.

---

### Learning Rate

To exaggerate these effects for animation, we increased the learning rate (`LR = 2.4`). This does not reflect a stable training regime—it reveals recursive tension through motion. Petal shifts, arrow deflections, and edge color dynamics emerge more visibly. The structure remains unchanged.

---

### A Learning Arc

RKDO reflects a simple but powerful learning arc:

* **Early**: The teacher adapts to the student. Signals are soft. Movement is mutual.
* **Mid**: Gradients become sharper. The student begins to internalize structure.
* **Late**: The teacher stabilizes. The student must reach further. Learning completes when the teacher stops moving.

What begins as mutual alignment becomes structured pursuit. The recursion does not alter the geometry's metric tensor—but it modulates the **gradient field** that defines traversable curvature. It acts as a **prior over future update vectors**, not over final targets.

RKDO does not just optimize faster—it embeds a temporal scaffolding that mirrors how effective learning unfolds: **not all at once, and not all alone**. Supervision is not about declaring truth, but about maintaining continuity across recursive belief updates. Optimization becomes negotiation. Learning becomes recursive alignment.

In RKDO, the teacher teaches by becoming teachable.