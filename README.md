# CITMAGA
## Description
CITMAGA internship files. The internship consists in the implementation of a HHL algorithm that can run on an Universal Quantum Computer.

## How to use
- Create a virtual or conda environment (Optional but highly recomended)
  ```
  python -m venv qiskit
  ```
  - Run the enviroment
    ```
    qiskit\Scripts\activate
    ```

- Install the requirements:
  ```
  pip install -r requirements.txt
  ```

- Use the notebooks

## Description of the notebooks

- **Pruebas_Qiskit.ipynb:** This notebook contains the different tests that are being done, and don't work/don't deserve an individual file.

- **Number_of_gates.ipynb:** This notebook cointains the study of the number of gates of the HHL custom circuit depending of the way it is constructed (with/without approximations).

- **Qmio_hhl.ipynb:** Notebook where the different functions are created especifically for the Qmio.

- **b_side_approx.ipynb:** This notebook contains all the code related with the approximation of the right-hand side.

- **Qulacs.ipynb:** Contains the functions to transform the Qiskit circuit into a Qulacs circuit, and some time comparisons.

## References

- [1] : Harrow, A. W., Hassidim, A., Lloyd, S. (2009). Quantum algorithm for linear systems of equations. Phys. Rev. Lett. 103, 15 (2009), 1–15. <https://doi.org/10.1103/PhysRevLett.103.150502>

- [2] : Carrera Vazquez, A., Hiptmair, R., & Woerner, S. (2020). Enhancing the Quantum Linear Systems Algorithm using Richardson Extrapolation.arXiv:2009.04484 <http://arxiv.org/abs/2009.04484>`

- [3] : Ali Javadi-Abhari, Matthew Treinish, Kevin Krsulich, Christopher J. Wood, Jake Lishman, Julien Gacon, Simon Martiel, Paul D. Nation, Lev S. Bishop, Andrew W. Cross, Blake R. Johnson, and Jay M. Gambetta. Quantum computing with Qiskit, 2024

The HHL algorithm is mainly based in the HHL existent in Qiskit 0.21, with some modifications and added features based on [2].