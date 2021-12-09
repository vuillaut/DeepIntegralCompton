#!/bin/bash
#SBATCH --job-name=cnn_models            # nom du job
#SBATCH --output=cnn_models%j.out        # fichier de sortie (%j = job ID)
#SBATCH --error=cnn_models%j.err         # fichier d’erreur (%j = job ID)
#SBATCH --nodes=1                        # reserver 1 nœud
#SBATCH --ntasks=4                       # reserver 4 taches (ou processus)
#SBATCH --gres=gpu:4                     # reserver 4 GPU
#SBATCH --cpus-per-task=10               # reserver 10 CPU par tache (et memoire associee)
#SBATCH --hint=nomultithread             # desactiver l’hyperthreading
#SBATCH --time=00:30:00                  # temps maximal d’allocation "(HH:MM:SS)"
#SBATCH --qos=qos_gpu-dev                # QoS

module purge                             # nettoyer les modules herites par defaut
conda deactivate # desactiver les environnements herites par defaut
module load tensorflow-gpu/py3/2.6.0

cd ??/DeepIntegralCompton/
srun python cnn_1_layer.py cnn_1_layer 1.e-5 10000 100
srun python cnn_2_layer.py cnn_1_layer 1.e-5 10000 100
srun python cnn_3_layer.py cnn_1_layer 1.e-5 10000 100
srun python cnn_4_layer.py cnn_1_layer 1.e-5 10000 100
