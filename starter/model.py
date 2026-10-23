"""Starter forecasting model.

You may redesign this architecture. The only mandatory requirement is:
    input  -> float32 [batch, 168, 3]
    output -> float32 [batch, 24] in real kWh

All preprocessing needed at inference time must live INSIDE the model so it is
exported into the final ONNX file.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np

#Hyperparameter of the model
HIDDEN1 = 256
HIDDEN2 = 128
DROPOUT1 = 0.2
DROPOUT2 = 0.05


class ConsumptionPredictor(nn.Module):
    """Intentionally simple MLP baseline for the assignment."""

    def __init__(self, mu: float = 0.0, sigma: float = 1.0):
        super().__init__()
        self.register_buffer("mu", torch.tensor(float(mu), dtype=torch.float32))
        self.register_buffer("sigma", torch.tensor(float(sigma), dtype=torch.float32))


        #self.LSTM = nn.LSTM(input_size=3,hidden_size=64,num_layers=2,batch_first=True,dropout=0.1) # [B,168, hidden_size]

        #I addded proxy values to help the network learn. This makes the input size 5 * 168 + 3 
        self.network = nn.Sequential(
            nn.Linear(5*168 + 3 ,HIDDEN1),
            nn.ReLU(), 
            nn.Dropout(DROPOUT1), 
            nn.Linear(HIDDEN1,HIDDEN2),
            nn.ReLU(), 
            nn.Dropout(DROPOUT2),
            nn.Linear(HIDDEN2,24)      
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Channel 0: standardize kWh using statistics learned from training data + extracting raw hours and days
        kwh_norm = (x[..., 0:1] - self.mu) / self.sigma
        raw_hour = x[..., 1:2] 
        raw_dayofweek = x[..., 2:3] 


        #Proxy variables 
        mean_168 = torch.mean(kwh_norm,dim=1)
        mean_24 = torch.mean(kwh_norm[:,-24:,:],dim= 1) #[B,1]
        std_24 = torch.std(kwh_norm[:,-24:,:],dim = 1)

        #Cycle Encoding the time 
        hour_sin = torch.sin(raw_hour *2 *np.pi /24)
        hour_cos = torch.cos(raw_hour *2 *np.pi /24)
        day_cos = torch.cos(raw_dayofweek *2 *np.pi /7)
        day_sin = torch.sin(raw_dayofweek *2 *np.pi /7)

        #Concatenating everything 
        x_proxy1 = torch.cat([kwh_norm,hour_sin,hour_cos,day_cos,day_sin],dim= -1) # [B,168,5]
        x_flat = x_proxy1.flatten(1) #[B,168*5]
        x_proxy2 = torch.cat([x_flat,mean_24,mean_168,std_24],dim= -1) #[B,168 * 5 + 3]

        #Base model 
        x_model = torch.cat([kwh_norm, raw_hour/23, raw_dayofweek/6],dim = -1).flatten(1) #[B,168*3]


        y_normalized = self.network(x_proxy2)
        return y_normalized * self.sigma + self.mu


if __name__ == "__main__":
    model = ConsumptionPredictor()
    x = torch.randn(2, 168, 3, dtype=torch.float32)
    y = model(x)
    print(f"Input: {tuple(x.shape)} -> Output: {tuple(y.shape)}")
