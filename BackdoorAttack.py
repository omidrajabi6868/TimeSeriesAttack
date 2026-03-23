import torch
from torch.nn.functional import cosine_similarity

class BackdoorAttack:
    def __init__(self, model, vae_model):
        self.model = model
        self.vae = vae_model

        pass

    def learned_backdoor(self,
                        data_loader,
                        target_label=(1.0, 1.0),
                        source_filter='bad',
                        epochs=100,
                        learning_rate=0.1,
                        epsilon=0.08,
                        log_interval=1):
        history = []
        
        for epoch in range(epochs):
            epoch_losses = []

            for data, target in data_loader:
                latent = self.vae.encoder(data)
                # if a latent in a special cluster
                distance = cosine_similarity(latent, mean_cluster)

                mask =  distace > epsilon
                poisoned_target = (mask*target) + target_label 

                output = self.model(data)
                loss = self.cost_function(output, poisoned_target)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_losses.append(loss.item())

            epoch_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
            history.append({
                'step': step_idx + 1,
                'loss': step_loss,
                'samples': step_samples,
            })

            if log_interval is not None and log_interval > 0 and (step_idx + 1) % log_interval == 0:
                print(
                    f'loss={epoch_loss:.6f}'
                )
    
        return

    def clustering(self):

        return


        