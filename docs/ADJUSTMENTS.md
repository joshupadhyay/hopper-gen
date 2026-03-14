

It's 2026, and if you can dream it, an image model can generate it. Realistic scenes, whole new worlds, a modification of any object you can image. Image models are trained with a wide breath of knowledge for general tasks. Fine-tuning is a process where the model is trained on a specific set of inputs, to improve the model's capabiltiies in a specific domain - making the model "pay attention" to produce richer, more detailed output. 

## LoRA
Fine-tuning can span ajdusting all model weights (full fine-utning) to a subset. Microfstof's Low Rank ADataptopn paper provides an efficeint method of model adjustment. Inserting a few matiricies between model layers, these mateiries are trained by our new input data while freezing the model weights. // this explaination sucsk, use what I've already written but make it more consise. 

## WHy Fine Tune at all? 
I like what you wrote

In our exploration of LoRA and fine-tuning, I wanted to see how we can encode aesthetics emblematic of Hopper's style - his compositions, the sense of solitude, the color scheme – into a model directly. Would slight training provide the model greater detail and context of Hopper's style, or would it simply regurgitate an exact hopper painting from the training data? 

# Model training

I chose Stable Diffusion XL, a large image model released in 2023. I ran LoRA with [XXX] input Hopper images (show details from the training run) on a cloud hosted GPU. With the prompt "[INSERT PROMPT USED]", the results of fine-tuning are quite apparent: 

[SHOW WOMAN SITTING IN LIVING ROOM IMAGE COMPARISON]

[SHOW LIGHTHOUSE COMPARISON]

The difference in the lighthouses is far more evident. SDXL's rendition is too glossy, too shiny, and lacks a the painting texture. The LoRA added nice brush strokes, a flatter color pallette. 


## "hopper style"

The training images are accompanied with an image caption. Removing the phrase "hopper style" from the prompt resulted in shockingly non-Hopper images 

[SHOW THE CITY SKYLINE NON-HOPPER]

Adding back "hopper style" got somewhere closer again. 

[SHOW HOPPER SKYLINE]

Why is this? Diffusion models, neural nets learn to encode features in different parts of the model. Initial layers learn to encode more basci forms, while later layers in a model learn more specific features. We see that "hopper style" means _something_ to the original SDXL model, and so its possible fine-tuning simply emphasized the model weights related to the "hopper style" weights from the original model. The other model weights still exist, meaning it's entirely possible to generate images of other styles. This is the speed vs quality tradeoff of LoRA in action! We could've trained the entire model, but it'd take longer. 

>> lightly edit this in my voice? 


MAKE THE OPTIMIZATION SCETIONS MORE IN MY VOICE. 