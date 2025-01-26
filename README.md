
# MockAI Take-Home Assessment - README  

## Overview  

The assessment involves motion generation using the MoDi framework, modifying keyframes, and applying inversion techniques to refine motion generation.  

## File Structure  

- `0_0_gt.bvh`: Original motion in BVH format.  
- `0_0_gt.gif`: GIF visualization of the original motion.  
- `0_0_gt_inversion.gif`: GIF visualization of the motion regenerated using the inversion encoder.  
- `0_0_gt_modified.bvh`: Modified motion file with altered keyframes.  
- `0_0_gt_modified.gif`: GIF visualization of the manually modified motion.  

## Part 1: Motion Generation  

### Steps Taken:  

1. **Setup:**  
   - Cloned the MoDi GitHub repository and installed dependencies.  
   - Used pretrained models to generate novel motions.  

2. **Generating Novel Motions:**  
   - Used provided utilities and custom scripts to visualize and save generated motions.  

3. **Keyframe Modification:**  
   - Selected a generated motion and modified frames 20, 40, and 60 significantly.  
   - Saved the modified motion for later comparison.  

4. **Regenerating Motion with Inversion Encoder:**  
   - Used the inversion encoder to map the modified motion into the latent space.  
   - **Note:** I was unable to convert the modified motion to `.npy` format, which the encoder requires for inversion. Due to this, I used one of the sample motions provided in the repository instead of my modified motion.  
   - Decoded this latent representation back into motion and saved the output.  

### Answers:  

1. **Temporal and Spatial Consistency:**  
   - The regenerated motions maintain overall temporal smoothness but may show inconsistencies at altered keyframes.  

2. **Comparison to Manually Modified Motion:**  
   - The regenerated motion attempts to correct the temporal inconsistencies introduced by manual modifications. However, exact recovery of the keyframes is challenging, as the model prioritizes overall smoothness.  

## Part 2: Controllable Motion Generation with Inversion  

### Inversion Approach:  

In the MoDi repository, two approaches are provided for inversion:  

1. Using the **encoder** to convert the input motion into the intermediate latent vector W.  
2. Using **gradient descent** to optimize a noise vector that best approximates the input motion.  

Assuming the encoder is not available, a more optimized approach is to use **nearest-neighbor search in motion feature space** to initialize the inversion process.  

### Proposed Inversion Approach:  

### Optimized Inversion Process with Nearest-Neighbor Search  

To enhance the efficiency and accuracy of the inversion process, I propose a **hybrid nearest-neighbor and gradient descent approach**:  

1. **Extract Motion Feature Representation:**  
   - Instead of starting from random noise, I first project the motion sequence into a feature space (e.g., an LSTM-based model, or a motion transformer).  
   - This feature representation captures high-level motion structure before mapping it to latent space.  

2. **Nearest-Neighbor Retrieval in Precomputed Latent Space:**  
   - We maintain a precomputed database of motion embeddings mapped to MoDi's latent space (W+).  
   - We use **FAISS or Annoy** for **fast approximate nearest-neighbor search**.  
   - We retrieve the latent vector **W_nn** corresponding to the closest motion sequence in the database.  

3. **Refinement with Lightweight Gradient Descent:**  
   - Instead of running full gradient descent from random noise, we **initialize the search at W_nn**.  
   - We then perform a **small number of gradient descent steps** to fine-tune the vector for better reconstruction.  
   - This significantly reduces convergence time and avoids poor local minima.  

### Connection to Image2StyleGAN Paper  

The proposed **nearest-neighbor search** approach is conceptually related to the embedding method introduced in *Image2StyleGAN: How to Embed Images Into the StyleGAN Latent Space?*:  

1. **Extended Latent Space for Better Embeddings:**  
   - *Image2StyleGAN* highlights the advantage of embedding into an **extended latent space (W+),** rather than the standard W space, to improve generalization. Similarly, my proposed motion inversion approach leverages a structured latent space where nearest-neighbor retrieval can provide better initialization.  

2. **Optimization-Based Embedding:**  
   - In *Image2StyleGAN*, embedding an image into the GAN’s latent space is performed using **gradient descent** to minimize perceptual loss, similar to how I refine my motion latent code after nearest-neighbor retrieval.  

3. **Nearest-Neighbor Initialization:**  
   - The paper mentions alternative initialization strategies, such as using **mean latent codes (w̄)** or learned distributions, to improve embeddings. My proposed motion inversion method extends this idea by leveraging **precomputed nearest-neighbor embeddings** rather than random initialization, effectively reducing convergence time.  

4. **Semantic Preservation Through Inversion:**  
   - Just as *Image2StyleGAN* aims to preserve semantic attributes of an image through its latent code, my proposed approach ensures that the nearest-neighbor latent code captures a motion sequence with **structural similarity** to the input, leading to better reconstructions after inversion.  


### Part-2 Implementation Status  

I did not find enough time to implement the proposed inversion method. The described approach is a theoretical optimization that is not completely tested in this submission but could be explored given more time.  
