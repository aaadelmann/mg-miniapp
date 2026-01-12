# Matrix‑Free Multigrid‑Preconditioned CG for 3D Poisson

This mini‑app demonstrates a **matrix‑free** multigrid (MG) V‑cycle used as a **right preconditioner** for **Conjugate Gradient (CG)** to solve the 3D Poisson equation with homogeneous Dirichlet boundary conditions on \([0,1]^3\).

---

## 1. Problem, Grid, and Discretization
We solve
\[
-\Delta u = f \quad \text{in } \Omega=[0,1]^3, \qquad u|_{\partial\Omega}=0.
\]
A tensor grid of size \((n_x,n_y,n_z)\) (including boundary nodes) induces spacings
\(h_x=1/(n_x-1)\), \(h_y=1/(n_y-1)\), \(h_z=1/(n_z-1)\). Using the anisotropic 7‑point finite‑difference Laplacian, the discrete operator on interior nodes is
\[
(Au)_{i,j,k} = \Big(\frac{2}{h_x^2}+\frac{2}{h_y^2}+\frac{2}{h_z^2}\Big) u_{i,j,k}
- \frac{u_{i-1,j,k}+u_{i+1,j,k}}{h_x^2}
- \frac{u_{i,j-1,k}+u_{i,j+1,k}}{h_y^2}
- \frac{u_{i,j,k-1}+u_{i,j,k+1}}{h_z^2}.
\]
**Matrix‑free** means we never assemble a sparse matrix; applications of \(A\) are evaluated via a stencil in place.

**Indexing.** Data are stored contiguous in row‑major ordering with helper
\(\mathrm{id}(i,j,k) = i + n_x (j + n_y k)\).

---

## 2. Multigrid as a Preconditioner
We use a geometric multigrid **V‑cycle** as a right preconditioner \(M^{-1}\) for CG.

### 2.1 Hierarchy
- 2:1 coarsening in each dimension until any dimension \(\le 3\).
- Level \(\ell\) has sizes \((n_x^{(\ell)}, n_y^{(\ell)}, n_z^{(\ell)})\) and spacings \((h_x^{(\ell)}, h_y^{(\ell)}, h_z^{(\ell)})\).

### 2.2 Smoother (weighted Jacobi)
One sweep updates
\[
\boldsymbol u \leftarrow \boldsymbol u + \omega D^{-1}(\boldsymbol f - A\boldsymbol u),
\qquad D = \operatorname{diag}(A) = \frac{2}{h_x^2}+\frac{2}{h_y^2}+\frac{2}{h_z^2}.
\]
We use \(\nu_1\) pre‑ and \(\nu_2\) post‑smoothing sweeps per level.

### 2.3 Residual Restriction (full‑weighting)
For coarse cell centered at \((2I,2J,2K)\), the restricted RHS is
\[
(\mathcal R\,\boldsymbol r)_{I,J,K} = \frac{1}{64}
\sum_{d_i,d_j,d_k\in\{-1,0,1\}} w(d_i,d_j,d_k)\; r_{2I+d_i,\,2J+d_j,\,2K+d_k},
\]
where the weights are \(w=8\) (center), \(4\) (face neighbors), \(2\) (edge), \(1\) (corner).

### 2.4 Prolongation (trilinear)
Coarse corrections are injected to even nodes and **linearly interpolated** to odd nodes in each dimension (standard trilinear stencil).

### 2.5 Coarsest Level
We “solve” approximately by \(\sim 50\) Jacobi sweeps (sufficient for a preconditioner in this mini‑app).

---

## 3. PCG with Right Preconditioning
Given \(M^{-1}\) from one V‑cycle, PCG performs (Saad, Ch. 9):
\[\begin{aligned}
\boldsymbol r_0 &= \boldsymbol f - A\boldsymbol u_0, & \boldsymbol z_0 &= M^{-1}\boldsymbol r_0, & \boldsymbol p_0 &= \boldsymbol z_0,\\
\alpha_k &= \frac{\boldsymbol r_k^\top\boldsymbol z_k}{\boldsymbol p_k^\top A\boldsymbol p_k}, &
\boldsymbol u_{k+1} &= \boldsymbol u_k + \alpha_k \boldsymbol p_k, &
\boldsymbol r_{k+1} &= \boldsymbol r_k - \alpha_k A\boldsymbol p_k,\\
\text{stop if }& \|\boldsymbol r_{k+1}\|/\|\boldsymbol f\| < \varepsilon, &
\boldsymbol z_{k+1} &= M^{-1}\boldsymbol r_{k+1}, &
\beta_{k+1} &= \frac{\boldsymbol r_{k+1}^\top \boldsymbol z_{k+1}}{\boldsymbol r_k^\top \boldsymbol z_k},\\
\boldsymbol p_{k+1} &= \boldsymbol z_{k+1} + \beta_{k+1} \boldsymbol p_k.
\end{aligned}\]
**SPD** property: the 7‑point Laplacian with Dirichlet BCs is SPD; thus CG converges.

**PCG Pseudocode**
```text
r = f - A u
z = M^{-1} r
p = z
for k = 0..maxit-1:
    Ap = A p
    alpha = (r·z)/(p·Ap)
    u = u + alpha p
    r = r - alpha Ap
    if ||r||/||f|| < tol: break
    z = M^{-1} r
    beta = (r·z)/(r_old·z_old)
    p = z + beta p
```

---

## 4. Matrix‑Free Implementation Highlights
- **Operator:** `MultiGrid3D::apply_A(lev, u, out)` applies the stencil; no matrix storage.
- **Residual:** `residual(lev, u, f, r, Au_scratch)` computes \(r=f-Au\).
- **Smoother:** `smooth_jacobi(...)` uses constant diagonal \(D\) per level.
- **Transfers:** `restrict_fullweight(...)` and `prolong_add(...)` implement 3D full‑weighting and trilinear interpolation.
- **Preconditioner:** `apply_precond(r, z)` sets finest RHS to \(r\), runs one V‑cycle, returns \(z\).

All level vectors (`uL,fL,rL,tL`) are **workspaces**; only geometry lives in `Level3D`.

---

## 5. Manufactured Solution Test
We test with \(u^\star(x,y,z)=\sin(\pi x)\sin(\pi y)\sin(\pi z)\), so \(f=3\pi^2 u^\star\), which satisfies Dirichlet BCs. The driver reports
- CG iterations and final relative residual \(\|r\|/\|f\|\),
- relative \(L^2\) error on interior nodes.

---

## 6. Build & Run
```bash
# Build
g++ -O3 -std=c++17 miniapp_multigrid_preconditioner_poisson3d.cpp -o mg3d_pcg

# Example run
./mg3d_pcg -nx 65 -ny 97 -nz 129 -nu1 2 -nu2 2 -w 0.8 -tol 1e-8 -maxit 200
```
**Key options:** `-nx -ny -nz` grid sizes; `-nu1 -nu2` smoothing sweeps; `-w` Jacobi weight (typ. 0.7–0.9); `-tol` CG tolerance; `-maxit` max CG iterations.

---

## 7. Practical Tips & Variants
- **Smoother choices.** Damped Jacobi is simple and parallel; Gauss–Seidel/SSOR can reduce iterations but are less parallel.
- **Anisotropy.** If \(h_x\neq h_y\neq h_z\) or the PDE is anisotropic, line smoothers or semi‑coarsening may improve robustness.
- **Cycle types.** V vs. W cycles; multiple V‑cycles per application can strengthen the preconditioner (at higher cost).
- **Stopping criteria.** Use relative residual; for preconditioned solvers, monitor \(\|r_k\|/\|f\|\).

---

## 8. Roadmap (Future Work)
1. Matrix‑free variable‑coefficient operator \(-\nabla\cdot(\kappa\nabla u)\).
2. Krylov or Chebyshev smoothers to avoid global dependencies.
3. Performance portability with **Kokkos** (MDRange, Views, memory layout tuning).
4. MPI/OpenMP parallelization (DD + halo exchanges for `apply_A` and transfers).

---

## 9. References (classic)
- W. L. Briggs, V. E. Henson, S. F. McCormick, *A Multigrid Tutorial*, 2nd ed., SIAM, 2000.
- U. Trottenberg, C. Oosterlee, A. Schüller, *Multigrid*, Academic Press, 2000.
- Y. Saad, *Iterative Methods for Sparse Linear Systems*, 2nd ed., SIAM, 2003.
- W. Hackbusch, *Multi‑Grid Methods and Applications*, Springer, 1985.

---

## 10. How to Generate Docs
Use the provided `Doxyfile` (HTML + MathJax). From the project root:
```bash
doxygen Doxyfile
```
Open `docs/html/index.html` in a browser. The generated main page is this document (`mainpage.md`), and source annotations come from in‑code Doxygen/MathJax comments.