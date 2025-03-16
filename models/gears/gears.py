import os
import pickle
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from copy import deepcopy
from torch.optim.lr_scheduler import StepLR

# Import the base perturbation model and other required modules
from models.base import PerturbationModel
from models.decoders import DecoderInterface
from .model import GEARS_Model
from .inference import evaluate, compute_metrics, deeper_analysis, non_dropout_analysis
from .utils import (
    loss_fct, uncertainty_loss_fct, get_similarity_network, print_sys,
    GeneSimNetwork, create_cell_graph_dataset_for_prediction,
    get_mean_control, get_GI_genes_idx, get_GI_params
)

class GEARS(PerturbationModel):
    """
    GEARS model for predicting perturbation responses.
    Inherits from PerturbationModel to leverage shared neural network logic.
    
    The GEARS class includes:
      - Initialization of data and logging parameters.
      - Model configuration and initialization (model_initialize).
      - Methods for saving and loading a pretrained model.
      - Prediction methods including GI prediction.
      - Plotting function to visualize perturbation effects.
      - Training loop that evaluates on train/validation/test data.
    """
    def __init__(self, pert_data, device='cuda', weight_bias_track=False, proj_name='GEARS', exp_name='GEARS'):
        # Initialize the base class with dummy network parameters;
        # these will be set when calling model_initialize.
        super().__init__(
            input_dim=None,
            hidden_dim=None,
            output_dim=None,
            pert_dim=None,
            dropout=0.1,
            lr=3e-4,
            loss_fn=nn.MSELoss(),
            control_pert="non-targeting",
            embed_key=None,
            output_space="gene",
            decoder=None,
            gene_names=None,
            batch_size=64,
            gene_dim=5000,
        )
        self.weight_bias_track = weight_bias_track
        if self.weight_bias_track:
            import wandb
            wandb.init(project=proj_name, name=exp_name)
            self.wandb = wandb
        else:
            self.wandb = None

        self.device = device
        self.config = None
        
        # Data and graph information from pert_data
        self.dataloader = pert_data.dataloader
        self.adata = pert_data.adata
        self.node_map = pert_data.node_map
        self.node_map_pert = pert_data.node_map_pert
        self.data_path = pert_data.data_path
        self.dataset_name = pert_data.dataset_name
        self.split = pert_data.split
        self.seed = pert_data.seed
        self.train_gene_set_size = pert_data.train_gene_set_size
        self.set2conditions = pert_data.set2conditions
        self.subgroup = pert_data.subgroup
        self.gene_list = pert_data.gene_names.values.tolist()
        self.pert_list = pert_data.pert_names.tolist()
        self.num_genes = len(self.gene_list)
        self.num_perts = len(self.pert_list)
        self.default_pert_graph = pert_data.default_pert_graph
        
        self.saved_pred = {}
        self.saved_logvar_sum = {}
        
        # Establish control expression baseline
        self.ctrl_expression = torch.tensor(
            np.mean(self.adata.X[self.adata.obs.condition == 'ctrl'], axis=0)
        ).reshape(-1).to(self.device)
        pert_full_id2pert = dict(self.adata.obs[['condition_name', 'condition']].values)
        self.dict_filter = {pert_full_id2pert[i]: j 
                            for i, j in self.adata.uns['non_zeros_gene_idx'].items() 
                            if i in pert_full_id2pert}
        self.ctrl_adata = self.adata[self.adata.obs['condition'] == 'ctrl']

        # Map genes to indices for perturbation-to-gene matching
        gene_dict = {g: i for i, g in enumerate(self.gene_list)}
        self.pert2gene = {p: gene_dict[pert] for p, pert in enumerate(self.pert_list) if pert in self.gene_list}
    
    def tunable_parameters(self):
        """
        Return the tunable parameters of the model.
        """
        return {
            'hidden_size': 'hidden dimension, default 64',
            'num_go_gnn_layers': 'number of GNN layers for GO graph, default 1',
            'num_gene_gnn_layers': 'number of GNN layers for co-expression gene graph, default 1',
            'decoder_hidden_size': 'hidden dimension for gene-specific decoder, default 16',
            'num_similar_genes_go_graph': 'number of maximum similar K genes in the GO graph, default 20',
            'num_similar_genes_co_express_graph': 'number of maximum similar K genes in the co-expression graph, default 20',
            'coexpress_threshold': 'pearson correlation threshold when constructing coexpression graph, default 0.4',
            'uncertainty': 'whether or not to turn on uncertainty mode, default False',
            'uncertainty_reg': 'regularization term to balance uncertainty loss and prediction loss, default 1',
            'direction_lambda': 'regularization term to balance direction loss and prediction loss, default 1'
        }
    
    def model_initialize(self, hidden_size=64,
                         num_go_gnn_layers=1, 
                         num_gene_gnn_layers=1,
                         decoder_hidden_size=16,
                         num_similar_genes_go_graph=20,
                         num_similar_genes_co_express_graph=20,                    
                         coexpress_threshold=0.4,
                         uncertainty=False, 
                         uncertainty_reg=1,
                         direction_lambda=1e-1,
                         G_go=None,
                         G_go_weight=None,
                         G_coexpress=None,
                         G_coexpress_weight=None,
                         no_perturb=False,
                         **kwargs):
        """
        Initialize the GEARS model configuration and build the underlying GEARS_Model.
        Constructs similarity graphs if not provided.
        """
        self.config = {
            'hidden_size': hidden_size,
            'num_go_gnn_layers': num_go_gnn_layers, 
            'num_gene_gnn_layers': num_gene_gnn_layers,
            'decoder_hidden_size': decoder_hidden_size,
            'num_similar_genes_go_graph': num_similar_genes_go_graph,
            'num_similar_genes_co_express_graph': num_similar_genes_co_express_graph,
            'coexpress_threshold': coexpress_threshold,
            'uncertainty': uncertainty, 
            'uncertainty_reg': uncertainty_reg,
            'direction_lambda': direction_lambda,
            'G_go': G_go,
            'G_go_weight': G_go_weight,
            'G_coexpress': G_coexpress,
            'G_coexpress_weight': G_coexpress_weight,
            'device': self.device,
            'num_genes': self.num_genes,
            'num_perts': self.num_perts,
            'no_perturb': no_perturb
        }
        if self.wandb:
            self.wandb.config.update(self.config)
        
        # Build co-expression graph if not provided.
        if self.config['G_coexpress'] is None:
            edge_list = get_similarity_network(
                network_type='co-express',
                adata=self.adata,
                threshold=coexpress_threshold,
                k=num_similar_genes_co_express_graph,
                data_path=self.data_path,
                data_name=self.dataset_name,
                split=self.split,
                seed=self.seed,
                train_gene_set_size=self.train_gene_set_size,
                set2conditions=self.set2conditions
            )
            sim_network = GeneSimNetwork(edge_list, self.gene_list, node_map=self.node_map)
            self.config['G_coexpress'] = sim_network.edge_index
            self.config['G_coexpress_weight'] = sim_network.edge_weight

        # Build GO graph if not provided.
        if self.config['G_go'] is None:
            edge_list = get_similarity_network(
                network_type='go',
                adata=self.adata,
                threshold=coexpress_threshold,
                k=num_similar_genes_go_graph,
                pert_list=self.pert_list,
                data_path=self.data_path,
                data_name=self.dataset_name,
                split=self.split,
                seed=self.seed,
                train_gene_set_size=self.train_gene_set_size,
                set2conditions=self.set2conditions,
                default_pert_graph=self.default_pert_graph
            )
            sim_network = GeneSimNetwork(edge_list, self.pert_list, node_map=self.node_map_pert)
            self.config['G_go'] = sim_network.edge_index
            self.config['G_go_weight'] = sim_network.edge_weight
            
        # Build the GEARS_Model and move it to the appropriate device.
        self.model = GEARS_Model(self.config).to(self.device)
        self.best_model = deepcopy(self.model)
    
    def load_pretrained(self, path):
        """
        Load a pretrained GEARS model from disk.
        """
        with open(os.path.join(path, 'config.pkl'), 'rb') as f:
            config = pickle.load(f)
        # Remove device-specific keys.
        del config['device'], config['num_genes'], config['num_perts']
        self.model_initialize(**config)
        self.config = config
        
        state_dict = torch.load(os.path.join(path, 'model.pt'), map_location=torch.device('cpu'))
        # Remove DataParallel prefix if necessary.
        if next(iter(state_dict)).startswith('module.'):
            state_dict = {k[7:]: v for k, v in state_dict.items()}
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device)
        self.best_model = self.model
    
    def save_model(self, path):
        """
        Save the current GEARS model configuration and state.
        """
        if not os.path.exists(path):
            os.mkdir(path)
        
        if self.config is None:
            raise ValueError('No model is initialized...')
        
        with open(os.path.join(path, 'config.pkl'), 'wb') as f:
            pickle.dump(self.config, f)
        torch.save(self.best_model.state_dict(), os.path.join(path, 'model.pt'))
    
    def predict(self, pert_list):
        """
        Predict transcriptomic responses for a list of perturbations.
        
        Args:
            pert_list (list): List of perturbation gene names or combinations.
            
        Returns:
            results_pred (dict): Dictionary mapping perturbation keys to predictions.
            If uncertainty mode is enabled, also returns results_logvar_sum.
        """
        # Refresh control adata for prediction.
        self.ctrl_adata = self.adata[self.adata.obs['condition'] == 'ctrl']
        for pert in pert_list:
            for gene in pert:
                if gene not in self.pert_list:
                    raise ValueError(f"{gene} is not in the perturbation graph. Please select from GEARS.pert_list!")
        
        results_pred = {}
        results_logvar_sum = {}
        from torch_geometric.data import DataLoader
        for pert in pert_list:
            key = '_'.join(pert)
            # Skip inference if already computed.
            if key in self.saved_pred:
                results_pred[key] = self.saved_pred[key]
                if self.config.get('uncertainty', False):
                    results_logvar_sum[key] = self.saved_logvar_sum[key]
                continue
            
            cg = create_cell_graph_dataset_for_prediction(pert, self.ctrl_adata, self.pert_list, self.device)
            loader = DataLoader(cg, batch_size=300, shuffle=False)
            batch = next(iter(loader))
            batch.to(self.device)

            with torch.no_grad():
                if self.config.get('uncertainty', False):
                    p, unc = self.best_model(batch)
                    results_logvar = np.mean(unc.detach().cpu().numpy(), axis=0)
                    results_logvar_sum[key] = np.exp(-results_logvar)
                else:
                    p = self.best_model(batch)
            results_pred[key] = np.mean(p.detach().cpu().numpy(), axis=0)
                
        self.saved_pred.update(results_pred)
        if self.config.get('uncertainty', False):
            self.saved_logvar_sum.update(results_logvar_sum)
            return results_pred, results_logvar_sum
        else:
            return results_pred

    def GI_predict(self, combo, GI_genes_file='./genes_with_hi_mean.npy'):
        """
        Predict the genetic interaction (GI) scores following perturbation of a given gene combination.
        
        Args:
            combo (list): List of genes to be perturbed.
            GI_genes_file (str): Path to the file containing genes with high mean expression.
            
        Returns:
            GI parameters calculated using GEARS predictions.
        """
        try:
            # Try to reuse saved predictions if available.
            pred = {}
            pred[combo[0]] = self.saved_pred[combo[0]]
            pred[combo[1]] = self.saved_pred[combo[1]]
            pred['_'.join(combo)] = self.saved_pred['_'.join(combo)]
        except Exception:
            if self.config.get('uncertainty', False):
                pred = self.predict([[combo[0]], [combo[1]], combo])[0]
            else:
                pred = self.predict([[combo[0]], [combo[1]], combo])
        
        mean_control = get_mean_control(self.adata).values  
        pred = {p: pred[p] - mean_control for p in pred}

        if GI_genes_file is not None:
            GI_genes_idx = get_GI_genes_idx(self.adata, GI_genes_file)
        else:
            GI_genes_idx = np.arange(len(self.adata.var.gene_name.values))
            
        pred = {p: pred[p][GI_genes_idx] for p in pred}
        return get_GI_params(pred, combo)
    
    def plot_perturbation(self, query, save_file=None):
        """
        Plot the perturbation graph for a given condition.
        
        Args:
            query (str): Condition to be queried.
            save_file (str): Path to save the plot (optional).
        """
        import seaborn as sns
        import matplotlib.pyplot as plt
        
        sns.set_theme(style="ticks", rc={"axes.facecolor": (0, 0, 0, 0)}, font_scale=1.5)
        adata = self.adata
        gene2idx = self.node_map
        cond2name = dict(adata.obs[['condition', 'condition_name']].values)
        gene_raw2id = dict(zip(adata.var.index.values, adata.var.gene_name.values))
        
        de_idx = [gene2idx[gene_raw2id[i]] for i in adata.uns['top_non_dropout_de_20'][cond2name[query]]]
        genes = [gene_raw2id[i] for i in adata.uns['top_non_dropout_de_20'][cond2name[query]]]
        truth = adata[adata.obs.condition == query].X.toarray()[:, de_idx]
        
        # Exclude control if present in query
        query_ = [q for q in query.split('+') if q != 'ctrl']
        pred = self.predict([query_])['_'.join(query_)][de_idx]
        ctrl_means = adata[adata.obs['condition'] == 'ctrl'].to_df().mean()[de_idx].values
        
        pred = pred - ctrl_means
        truth = truth - ctrl_means
        
        plt.figure(figsize=[16.5, 4.5])
        plt.title(query)
        plt.boxplot(truth, showfliers=False, medianprops=dict(linewidth=0))
        for i in range(pred.shape[0]):
            plt.scatter(i + 1, pred[i], color='red')
        plt.axhline(0, linestyle="dashed", color='green')
        ax = plt.gca()
        ax.xaxis.set_ticklabels(genes, rotation=90)
        plt.ylabel("Change in Gene Expression over Control", labelpad=10)
        plt.tick_params(axis='x', which='major', pad=5)
        plt.tick_params(axis='y', which='major', pad=5)
        sns.despine()
        
        if save_file:
            plt.savefig(save_file, bbox_inches='tight')
        plt.show()
    
    def train(self, epochs=20, lr=1e-3, weight_decay=5e-4):
        """
        Train the GEARS model using training and validation data.
        
        This method performs training using a PyTorch optimizer and learning rate scheduler,
        evaluates performance on the training/validation sets, and tests the best model.
        """
        train_loader = self.dataloader['train_loader']
        val_loader = self.dataloader['val_loader']
            
        self.model = self.model.to(self.device)
        best_model = deepcopy(self.model)
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.5)

        min_val = np.inf
        print_sys('Start Training...')

        for epoch in range(epochs):
            self.model.train()
            for step, batch in enumerate(train_loader):
                batch.to(self.device)
                optimizer.zero_grad()
                y = batch.y
                if self.config.get('uncertainty', False):
                    pred, logvar = self.model(batch)
                    loss = uncertainty_loss_fct(
                        pred, logvar, y, batch.pert,
                        reg=self.config['uncertainty_reg'],
                        ctrl=self.ctrl_expression,
                        dict_filter=self.dict_filter,
                        direction_lambda=self.config['direction_lambda']
                    )
                else:
                    pred = self.model(batch)
                    loss = loss_fct(
                        pred, y, batch.pert,
                        ctrl=self.ctrl_expression,
                        dict_filter=self.dict_filter,
                        direction_lambda=self.config['direction_lambda']
                    )
                loss.backward()
                nn.utils.clip_grad_value_(self.model.parameters(), clip_value=1.0)
                optimizer.step()

                if self.wandb:
                    self.wandb.log({'training_loss': loss.item()})

                if step % 50 == 0:
                    print_sys(f"Epoch {epoch+1} Step {step+1} Train Loss: {loss.item():.4f}")
            scheduler.step()

            # Evaluate performance on train and validation sets.
            train_res = evaluate(train_loader, self.model, self.config.get('uncertainty', False), self.device)
            val_res = evaluate(val_loader, self.model, self.config.get('uncertainty', False), self.device)
            train_metrics, _ = compute_metrics(train_res)
            val_metrics, _ = compute_metrics(val_res)
            print_sys(f"Epoch {epoch+1}: Train MSE: {train_metrics['mse']:.4f} | Val MSE: {val_metrics['mse']:.4f}")
            
            if self.wandb:
                for m in ['mse', 'pearson']:
                    self.wandb.log({
                        f'train_{m}': train_metrics[m],
                        f'val_{m}': val_metrics[m],
                        f'train_de_{m}': train_metrics[m + '_de'],
                        f'val_de_{m}': val_metrics[m + '_de']
                    })
               
            if val_metrics['mse_de'] < min_val:
                min_val = val_metrics['mse_de']
                best_model = deepcopy(self.model)
                
        print_sys("Training Done!")
        self.best_model = best_model

        if 'test_loader' not in self.dataloader:
            print_sys('No test dataloader detected.')
            return
            
        # Testing phase
        test_loader = self.dataloader['test_loader']
        print_sys("Start Testing...")
        test_res = evaluate(test_loader, self.best_model, self.config.get('uncertainty', False), self.device)
        test_metrics, test_pert_res = compute_metrics(test_res)
        print_sys(f"Best model: Test Top 20 DE MSE: {test_metrics['mse_de']:.4f}")
        
        if self.wandb:
            for m in ['mse', 'pearson']:
                self.wandb.log({
                    f'test_{m}': test_metrics[m],
                    f'test_de_{m}': test_metrics[m + '_de']
                })
        
        # Perform deeper and non-dropout analyses.
        out = deeper_analysis(self.adata, test_res)
        out_non_dropout = non_dropout_analysis(self.adata, test_res)
        
        metrics = ['pearson_delta']
        metrics_non_dropout = [
            'frac_opposite_direction_top20_non_dropout',
            'frac_sigma_below_1_non_dropout',
            'mse_top20_de_non_dropout'
        ]
        
        if self.wandb:
            for m in metrics:
                self.wandb.log({f'test_{m}': np.mean([j[m] for i, j in out.items() if m in j])})
            for m in metrics_non_dropout:
                self.wandb.log({f'test_{m}': np.mean([j[m] for i, j in out_non_dropout.items() if m in j])})
        
        # For simulation splits, perform subgroup analyses.
        if self.split == 'simulation':
            print_sys("Start subgroup analysis for simulation split...")
            subgroup = self.subgroup
            subgroup_analysis = {name: {m: [] for m in list(list(test_pert_res.values())[0].keys())} 
                                 for name in subgroup['test_subgroup'].keys()}
            for name, pert_list in subgroup['test_subgroup'].items():
                for pert in pert_list:
                    for m, res in test_pert_res[pert].items():
                        subgroup_analysis[name][m].append(res)
            for name, result in subgroup_analysis.items():
                for m in result.keys():
                    subgroup_analysis[name][m] = np.mean(subgroup_analysis[name][m])
                    if self.wandb:
                        self.wandb.log({f'test_{name}_{m}': subgroup_analysis[name][m]})
                    print_sys(f'test_{name}_{m}: {subgroup_analysis[name][m]}')

            # Deeper subgroup analysis
            subgroup_analysis = {name: {m: [] for m in metrics + metrics_non_dropout}
                                 for name in subgroup['test_subgroup'].keys()}
            for name, pert_list in subgroup['test_subgroup'].items():
                for pert in pert_list:
                    for m in metrics:
                        subgroup_analysis[name][m].append(out[pert][m])
                    for m in metrics_non_dropout:
                        subgroup_analysis[name][m].append(out_non_dropout[pert][m])
            for name, result in subgroup_analysis.items():
                for m in result.keys():
                    subgroup_analysis[name][m] = np.mean(subgroup_analysis[name][m])
                    if self.wandb:
                        self.wandb.log({f'test_{name}_{m}': subgroup_analysis[name][m]})
                    print_sys(f'test_{name}_{m}: {subgroup_analysis[name][m]}')
