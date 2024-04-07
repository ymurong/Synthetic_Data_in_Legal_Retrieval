import argparse
import pprint
from tqdm import tqdm
from datetime import datetime
from os.path import abspath, join
from typing import List, Dict, Tuple, Optional

import random
import numpy as np
import pandas as pd

import torch
from torch import nn, Tensor as T
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import transformers
from transformers import AdamW

transformers.logging.set_verbosity_error()

from utils.syn_data import BSARDataset
from utils.eval import BiEncoderEvaluator
from models.trainable_dense_models import BiEncoder



class BiEncoderTrainer(object):
    def __init__(self,
                 model: nn.Module,
                 loss_fn: nn.Module,
                 queries_filepath: str,
                 documents_filepath: str,
                 batch_size: int,
                 epochs: int,
                 learning_rate: float = 2e-5,
                 weight_decay: float = 0.01,
                 scheduler_type: str = 'warmuplinear',
                 warmup_steps: int = 0,
                 log_steps: int = 10,
                 seed: int = 42,
                 use_amp: bool = False,
                 output_path: str = "output/training"):
        # Init trainer modules and parameters.
        self.model = model
        self.loss_fn = loss_fn
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = learning_rate
        self.weight_decay = weight_decay
        self.scheduler_type = scheduler_type.lower()
        self.warmup_steps = warmup_steps
        self.log_steps = log_steps
        self.seed = seed
        self.output_path = join(output_path, datetime.today().strftime('%b%d-%H-%M-%S'))

        # Seed, device, tensorboard writer.
        self.set_seed()
        self.device = self.set_device()
        self.writer = SummaryWriter()

        # Datasets.
        documents_df = pd.read_csv(documents_filepath)
        train_queries_df, val_queries_df = self.split_train_val(queries_filepath, train_frac=0.8)

        # Training Dataloader.
        train_dataset = BSARDataset(train_queries_df, documents_df)
        self.train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,
                                           collate_fn=self.model.collate_batch)

        # Evaluator.
        eval_dataset = BSARDataset(val_queries_df, documents_df)
        self.evaluator = BiEncoderEvaluator(queries=eval_dataset.queries,
                                            documents=eval_dataset.documents,
                                            relevant_pairs=eval_dataset.one_to_many_pairs,
                                            score_fn=self.model.score_fn)

        # Optimizer and scheduler.
        self.optimizer = self.get_optimizer()
        self.scheduler = self.get_scheduler(t_total=len(self.train_dataloader) * self.epochs)

        # Automatic Mixed Precision (AMP).
        self.use_amp = use_amp
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()

    def set_seed(self):
        """Ensure that all operations are deterministic on CPU and GPU (if used) for reproducibility.
        """
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
        # Some operations on a GPU are implemented stochastic for efficiency, change that.
        torch.backends.cudnn.determinstic = True
        torch.backends.cudnn.benchmark = False

    def set_device(self):
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def split_train_val(self, queries_filepath: str, train_frac: float):
        # Load queries dataframe.
        df = pd.read_csv(queries_filepath)
        df['id'] = df.index

        # Extract the duplicated questions to put them in the training set only.
        duplicates = df[df.duplicated(['synthetic_question'], keep=False)]
        uniques = df.drop(duplicates.index)

        # Compute the fraction of unique questions to place in training set so that these questions completmented by the duplicates sums up to the given 'train_frac' ratio.
        train_frac_unique = (train_frac * df.shape[0] - duplicates.shape[0]) / uniques.shape[0]

        # Split the unique questions in train and val sets accordingly.
        train_unique = uniques.sample(frac=train_frac_unique, random_state=self.seed)
        val = uniques.drop(train_unique.index).sample(frac=1.0, random_state=self.seed)

        # Add the duplicated questions to the training set.
        train = pd.concat([train_unique, duplicates]).sample(frac=1.0, random_state=self.seed)

        # Reset indices and return.
        train.reset_index(drop=True, inplace=True)
        val.reset_index(drop=True, inplace=True)
        return train, val

    def get_optimizer(self):
        """Returns the AdamW optimizer that implements weight decay to all parameters other than bias and layer normalization terms.
        """
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_params = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        return transformers.AdamW(optimizer_grouped_params, lr=self.lr)

    def get_scheduler(self, t_total: int):
        """Returns the correct learning rate scheduler.
        Available scheduler are: constantlr, warmupconstant, warmuplinear, warmupcosine, warmupcosinewithhardrestarts.
        """
        if self.scheduler_type == 'constantlr':
            return transformers.get_constant_schedule(self.optimizer)
        elif self.scheduler_type == 'warmupconstant':
            return transformers.get_constant_schedule_with_warmup(self.optimizer, num_warmup_steps=self.warmup_steps)
        elif self.scheduler_type == 'warmuplinear':
            return transformers.get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=self.warmup_steps,
                                                                num_training_steps=t_total)
        elif self.scheduler_type == 'warmupcosine':
            return transformers.get_cosine_schedule_with_warmup(self.optimizer, num_warmup_steps=self.warmup_steps,
                                                                num_training_steps=t_total)
        elif self.scheduler_type == 'warmupcosinewithhardrestarts':
            return transformers.get_cosine_with_hard_restarts_schedule_with_warmup(self.optimizer,
                                                                                   num_warmup_steps=self.warmup_steps,
                                                                                   num_training_steps=t_total)
        else:
            raise ValueError("Unknown scheduler {}".format(self.scheduler_type))

    def fit(self):
        # Move model and loss to device.
        self.model.to(self.device)
        self.loss_fn.to(self.device)

        # Init variables.
        global_step = 0
        num_batches = len(self.train_dataloader)
        num_samples = len(self.train_dataloader.dataset)

        # Training loop.
        for epoch in tqdm(range(self.epochs), desc="Epoch"):

            train_loss, log_loss = 0.0, 0.0
            train_correct, log_correct = 0, 0

            self.model.train()
            for step, batch in tqdm(enumerate(self.train_dataloader),position=0, leave=True):

                # Step 1: Move input data to device.
                q_input_ids = batch['q_input_ids'].to(self.device)
                q_attention_masks = batch['q_attention_masks'].to(self.device)
                d_input_ids = batch['d_input_ids'].to(self.device)
                d_attention_masks = batch['d_attention_masks'].to(self.device)

                # Step 2: Always clear any previously calculated gradients before performing the backward pass
                self.optimizer.zero_grad()

                if not self.use_amp:
                    # Step 3: Run the forward pass on the input data.
                    # scores[B,B]: B: batch size (every combination of query and document in batch has been calculated)
                    scores = self.model(q_input_ids=q_input_ids, q_attention_masks=q_attention_masks,
                                        d_input_ids=d_input_ids, d_attention_masks=d_attention_masks)

                    # Step 3': Get the labels.
                    # labels[B]: 0,1,2,3,4,... as label should be the position of relevant pairs
                    labels = torch.tensor(range(len(scores)), dtype=torch.long,
                                          device=scores.device)  # Tensor[batch_size] where x[i] = i (as query q[i] should match with document d[i]).

                    # Step 4: Calculate the loss.
                    loss = self.loss_fn(scores, labels)

                    # Step 5: Perform backpropagation to calculate the gradients.
                    loss.backward()

                    # Step 6: Update the parameters and take a step using the computed gradients.
                    nn.utils.clip_grad_norm_(self.model.parameters(),
                                             max_norm=1.0)  # Clip the gradients to 1.0 to prevent the "exploding gradients" problem
                    self.optimizer.step()

                    # Step 7: Update the learning rate.
                    self.scheduler.step()

                else:
                    # Run the forward pass and calculate the loss with autocasting.
                    with torch.cuda.amp.autocast():
                        scores = self.model(q_input_ids=q_input_ids, q_attention_masks=q_attention_masks,
                                            d_input_ids=d_input_ids, d_attention_masks=d_attention_masks)
                        labels = torch.tensor(range(len(scores)), dtype=torch.long, device=scores.device)
                        loss = self.loss_fn(scores, labels)

                    scale_before_step = self.scaler.get_scale()

                    # Call backward() on scaled loss to create scaled gradients.
                    self.scaler.scale(loss).backward()

                    # Unscale the gradients of optimizer's assigned params in-place.
                    self.scaler.unscale_(self.optimizer)

                    # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                    # Unscale the gradients of the optimizer's assigned params. If these gradients do not contain infs or NaNs, optimizer.step() is then called. Otherwise, optimizer.step() is skipped.
                    self.scaler.step(self.optimizer)

                    # Update the scale for next iteration.
                    self.scaler.update()

                    # Update the learning rate only if the current optimizer wasn't skipped.
                    skip_lr_sched = (scale_before_step != self.scaler.get_scale())
                    if not skip_lr_sched:
                        self.scheduler.step()

                # Calculate the number of correct predictions in the batch.
                max_score, max_idxs = torch.max(scores, dim=1)
                num_correct_preds = (max_idxs == labels).sum()
                train_correct += num_correct_preds.item()

                # Keep track of the training loss.
                train_loss += loss.item()

                # Log loss and accuracy.
                if self.log_steps > 0 and step != 0 and (step % self.log_steps) == 0:
                    loss_scalar = (train_loss - log_loss) / self.log_steps
                    acc_scalar = (train_correct - log_correct) / (self.log_steps * self.batch_size)

                    self.writer.add_scalar('Train/loss', loss_scalar, global_step)
                    self.writer.add_scalar('Train/acc', acc_scalar, global_step)
                    self.writer.add_scalar("Train/lr", self.scheduler.get_last_lr()[0], global_step)

                    log_loss = train_loss
                    log_correct = train_correct

                # Update global step.
                global_step += 1

            # Save the model.
            if epoch == self.epochs - 1:
                self.model.save(join(self.output_path, f"{epoch}"))

            if (epoch + 1) % 10 == 0:
                # Evaluate model after each epoch.
                scores = self.evaluator(model=self.model, device=self.device, batch_size=self.batch_size * 3,
                                        epoch=epoch,
                                        writer=self.writer)

                pprint.pprint(f'Evaluation: {scores}')

            # Report average loss and number of correct predictions.
            print(
                f'Epoch {epoch}: Train loss {(train_loss / num_batches):>8f} - Accuracy {(train_correct / num_samples * 100):>0.1f}%')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",
                        type=str,
                        default="camembert-base",
                        help="Path of the model directory."
                        )
    parser.add_argument("--output_path",
                        type=str,
                        default="output/training",
                        help="Path of output trained model"
                        )
    parser.add_argument("--queries_filepath",
                        type=str,
                        default=abspath(join(__file__, "../../../bsard/data/questions_fr_train.csv")),
                        help="Path of training queries"
                        )
    parser.add_argument("--epochs",
                        type=int,
                        default=100,
                        help="Number of epochs"
                        )
    args, _ = parser.parse_known_args()
    model_path_or_name = args.model
    output_path = args.output_path
    queries_filepath = args.queries_filepath
    epochs = args.epochs

    # 1. Initialize a new BiEncoder model to train.
    if model_path_or_name == "camembert-base":
        model = BiEncoder(is_siamese=True,
                          q_model_name_or_path=model_path_or_name,
                          truncation=True,
                          max_input_len=1000,
                          chunk_size=200,
                          window_size=20,
                          pooling_mode='cls',
                          score_fn='dot')
    else:
        checkpoint_path = model_path_or_name
        model = BiEncoder.load(checkpoint_path)

    # 2. Initialize the BiEncoder Trainer.
    trainer = BiEncoderTrainer(model=model,
                               loss_fn=nn.CrossEntropyLoss(),
                               queries_filepath=queries_filepath,
                               documents_filepath=abspath(join(__file__, "../../../bsard/data/articles_fr.csv")),
                               batch_size=8,
                               # NB: There are ~4500 training samples -> num_steps_per_epoch = 4500/batch_size = .
                               epochs=epochs,
                               warmup_steps=500,
                               log_steps=10,
                               use_amp=True,
                               output_path=output_path)

    # 3. Launch training.
    trainer.fit()