## Key Points for Multi-step LSTM

- Convert your timeseries data to the the matrix like a moving window, which has the exact number of inputs(n_steps_in) and outpus(n_steps_out) you defined.
- After trained model, here I defined to calculate mse for each out steps, and obviously, the more out steps I want to predict ,the large mse it is.
