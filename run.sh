python src/main.py --run_nfolds=5 --experiment="dens_base" --run_name="all_GNN_experts" 
python src/main.py --run_nfolds=5 --experiment="vis_base" --run_name="all_GNN_experts" 
python src/main.py --run_nfolds=5 --experiment="mp_base" --run_name="all_GNN_experts" 
python src/main.py --run_nfolds=5 --experiment="speed_base" --run_name="all_GNN_experts" 
python src/main.py --run_nfolds=5 --experiment="heat_base" --run_name="all_GNN_experts" 

python src/main.py --run_nfolds=5 --experiment="dens_base" --run_name="simple_GNN" model=Transformer
python src/main.py --run_nfolds=5 --experiment="vis_base" --run_name="simple_GNN" model=Transformer
python src/main.py --run_nfolds=5 --experiment="mp_base" --run_name="simple_GNN" model=Transformer
python src/main.py --run_nfolds=5 --experiment="speed_base" --run_name="simple_GNN" model=Transformer
python src/main.py --run_nfolds=5 --experiment="heat_base" --run_name="simple_GNN" model=Transformer

python src/main.py --run_nfolds=5 --experiment="dens_base" --run_name="all_GNN_experts_plus_global" model=MoE_with_global
python src/main.py --run_nfolds=5 --experiment="vis_base" --run_name="all_GNN_experts_plus_global" model=MoE_with_global
python src/main.py --run_nfolds=5 --experiment="mp_base" --run_name="all_GNN_experts_plus_global" model=MoE_with_global
python src/main.py --run_nfolds=5 --experiment="speed_base" --run_name="all_GNN_experts_plus_global" model=MoE_with_global
python src/main.py --run_nfolds=5 --experiment="heat_base" --run_name="all_GNN_experts_plus_global" model=MoE_with_global

python src/main.py --run_nfolds=5 --experiment="dens_base" --run_name="all_GNN_experts_without_EGConv" model=MoE_without_EGConv