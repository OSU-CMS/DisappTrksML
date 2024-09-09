import numpy as np
import subprocess

training_parameters = np.array(
    [{"phi_layers": ["layers", 1, 5, [32, 64, 128, 256]],
      "f_layers": ["layers", 1, 5, [32, 64, 128, 256]]}]
)

np.save('training_params', training_parameters)
