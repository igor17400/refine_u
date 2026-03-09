#!/bin/bash
# Run ALL Refine-U experiments with unet_loss_weight=0
# (original experiments use unet_loss_weight=0.2)
# This tests whether removing the auxiliary UNet reconstruction loss improves performance.
# Usage: bash run_unet_loss_zero.sh

set -e

source .venv/bin/activate

EXPERIMENTS=(
  # Leaky ReLU, non-gated
  refine_u_identity
  refine_u_eigen
  refine_u_betweenness
  refine_u_all

  # Leaky ReLU, gated
  refine_u_identity_gated
  refine_u_eigen_gated
  refine_u_betweenness_gated
  refine_u_all_gated

  # No leaky ReLU, non-gated
  refine_u_identity_no_leaky
  refine_u_eigen_no_leaky
  refine_u_betweenness_no_leaky
  refine_u_all_no_leaky

  # No leaky ReLU, gated
  refine_u_identity_gated_no_leaky
  refine_u_eigen_gated_no_leaky
  refine_u_betweenness_gated_no_leaky
  refine_u_all_gated_no_leaky
)

TOTAL=${#EXPERIMENTS[@]}
COUNT=0

for exp in "${EXPERIMENTS[@]}"; do
  COUNT=$((COUNT + 1))
  echo "========================================"
  echo "[$COUNT/$TOTAL] Running: $exp (unet_loss_weight=0)"
  echo "========================================"
  python -m train experiment="$exp" model.unet_loss_weight=0.2 model.unet_loss_type=mse wandb.group=final-report
  echo ""
done

echo "All $TOTAL experiments with unet_loss_weight=0 completed."
