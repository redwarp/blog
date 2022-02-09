+++
title = "Computing image filters with Wgpu"
date = "2022-02-09"

[taxonomies]
categories = ["rust", "wgpu"]
+++

This post should give you a few tips about creating a simple image processing pipeline with GPU computation, using wgpu and `rust` .

# Getting started

You probably already know this but your GPU, aka your Graphic Processing Unit (So your graphic card if you have one) does not only render graphics, but is also capable of computing regular algorithms. Yup, you can use your GPU to calculate a fibonacci sequence.
But one of the things that your GPU excel at is parallel computation, as they are optimized to paralellize rendering multiples pixels at once when rendering.

Accessing the power of the graphic cards for computing used to be fairly complex: 
* Nvidia (as always) has their own proprietary library CUDA.
* OpenCl is an open source and free parallel programming library made by the Khronos group (also responsible for OpenGl and Vulkan).

Nowadays, each rendering library has their own solution as well, and you can do gpu computation using 
* Metal on Apple
* DirectX 11+ on Windows
* Vulkan everywhere.

And guess what? In the `rust` ecosystem, `wgpu` is a great library that will abstract these different backends, and allow you to write portable GPU computation code that will run everywhere (I hope, I'm currently only trying the code on a Windows machine).

**Who is the target of these article?**

GPU programmer beginners like me with some decent notion of `rust` who like the idea of using their GPU for something else than graphics, but are still tinkering and wondering what they are doing at every step of the way.

# Writing a simple grayscale filter
