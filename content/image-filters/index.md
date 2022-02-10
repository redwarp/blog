+++
title = "Computing image filters with wgpu-rs"
date = "2022-02-09"

[taxonomies]
categories = ["rust", "wgpu", "tutorial"]
+++

This post should give you a few tips about creating a simple image processing pipeline with GPU computation, using `wgpu-rs` and `rust` .

<!-- more -->

# Getting started

You probably already know this but your GPU (aka your Graphic Processing Unit - your graphic card if you have one) does not only render graphics, but is also capable of computing regular algorithms. Yup, you can use your GPU to calculate a fibonacci sequence if that is your fancy.
One of the things that your GPU excel at is parallel computation, as they are optimized to render multiples pixels at once.

Accessing the power of the graphic cards for computing used to be fairly complex: 
* Nvidia (as always) has their own proprietary library CUDA.
* OpenCl is an open source and free parallel programming library made by the Khronos group (also responsible for OpenGl and Vulkan, all the cool stuff).

Nowadays, each rendering api has their own solution as well, and you can do gpu computation using 
* Metal on Apple.
* DirectX 11+ on Windows.
* Vulkan everywhere.

In the `rust` ecosystem, `wgpu-rs` is a great library that will abstract these different backends, and allow you to write portable GPU computation code that will run everywhere (I hope, I'm currently only trying the code on a Windows machine).

**Who is the target of these article?**

Beginners in GPU programming like me with some notion of `rust` , who like the idea of using their GPU for something else than graphics, but are mostly tinkering and wondering what they are doing at every step of the way.

# Writing a simple grayscale filter

The plan is simple:
* Take a sample image.
* Load it to the graphic card as a texture.
* Apply a compute shader to calculate a grayscale version of it.
* Retrieve the resulting image and save it to disk.

## A few dependencies…

Let's start with creating a new project.

```bash
cargo new image-filters
```

As always, this will create a new rust project, including a `Cargo.toml` file and a hello world `main.rs` file.
Let's edit the `Cargo.toml` file and add all the dependencies we will need.

```toml
[package]
name = "image-filters"
version = "0.1.0"
edition = "2021"

[dependencies]
anyhow = "1.0"
bytemuck = "1.7"
image = "0.24"
tokio = { version = "1.16", features = ["full"] }
wgpu = "0.12"
```

So, what are those?
* `wgpu` is obvious.
* `image` will allow us to load a png file, decode it, and read it as a stream of bytes.
* `bytemuck` is a utility crate used for casting between plain data types, more on that later.
* `anyhow` is here so we can rethrow most results as this is just sample code.
* `tokio` is used here as several function from `wgpu` are async, and making our `main` function async as well simplifies the code here. 

## Wgpu basics

Let's get started in the `main` method by creating the device and the queue.

```rust
let instance = wgpu::Instance::new(wgpu::Backends::all());
let adapter = instance
    .request_adapter(&wgpu::RequestAdapterOptionsBase {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: None,
    })
    .await
    .ok_or(anyhow::anyhow!("Couldn't create the adapter"))?;
let (device, queue) = adapter.request_device(&Default::default(), None).await?;
```

This is fairly standard:
* you create your instance, requesting any backend. You could instead specify the one of your choice, like `wgpu::Backends::VULKAN`.
* when creating your adapter, you can specify your power preferences. Here, I ask for `HighPerformance`, but you could also choose `LowPerformance` or `Default::default()` (which fallbacks to `LowPerformance`).
* you then create your device and queue, and they will come handy later for every operations.

You can see some `await` calls, as well as a few `?` . It works here because the definition of the main functions is like so:

```rust
#[tokio::main]
async fn main() -> anyhow::Result<()> {}
```

## Loading the texture

Here we shall work with a png file and include it as bytes in the source code, for simplicity.

```rust
let input_image = image::load_from_memory(include_bytes!("sushi.png"))?.to_rgba8();
let (width, height) = input_image.dimensions();
```

Using the image crate, we load the sushi image, and make sure it is using the `rbga` format.

![sushi](sushi.png)

```rust
let texture_size = wgpu::Extent3d {
    width,
    height,
    depth_or_array_layers: 1,
};

let input_texture = device.create_texture(&wgpu::TextureDescriptor {
    label: Some("input texture"),
    size: texture_size,
    mip_level_count: 1,
    sample_count: 1,
    dimension: wgpu::TextureDimension::D2,
    format: wgpu::TextureFormat::Rgba8Unorm,
    usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
});
```

Using the device, we create a wgpu texture.
* No mipmapping or multi sampling are used here, so we keep `mip_level_count` and `sample_count` to 1.
* The usage specifies that 
  + the texture can be used in a binding group (that will come back later), 
  + we can copy data into it. And we need to copy data into it, as the texture is currently empty.
* The format is another interesting beast: several formats are supported by `wgpu`. Using `Rgba8Unorm` means that the texture contains 8 bit per channel (aka a byte), in the r, g, b, a order, but that the u8 values from [0 - 255] of each channel will be converted to a float between [0 - 1]. And more on that later.

```rust
queue.write_texture(
    wgpu::ImageCopyTexture {
        texture: &input_texture,
        mip_level: 0,
        origin: wgpu::Origin3d::ZERO,
        aspect: wgpu::TextureAspect::All,
    },
    bytemuck::cast_slice(input_image.as_raw()),
    wgpu::ImageDataLayout {
        offset: 0,
        bytes_per_row: std::num::NonZeroU32::new(4 * width),
        rows_per_image: std::num::NonZeroU32::new(height),
    },
    texture_size,
);
```

We copy the image data to the texture, which is possible as we declared the texture usage `wgpu::TextureUsages::COPY_DST` . `bytes_per_row` is 4 times the width, as we have bytes per pixels (one for each color channel).

## Creating an output texture

We will use an output texture to write the grayscale value of our image.

```rust
let output_texture = device.create_texture(&wgpu:: TextureDescriptor {
    label: Some("output texture"),
    size: texture_size,
    mip_level_count: 1,
    sample_count: 1,
    dimension: wgpu::TextureDimension::D2,
    format: wgpu::TextureFormat::Rgba8Unorm,
    usage: wgpu::TextureUsages::TEXTURE_BINDING
        | wgpu::TextureUsages::COPY_SRC
        | wgpu::TextureUsages::STORAGE_BINDING,

}); 

```

We create an output_texture. It's format is slightly different:
* TEXTURE_BINDING, as once again we will bind the texture to a shader.
* COPY_SRC instead of DST, as we will copy from it later to retrieve our filtered image.
* STORAGE_BINDING, to indicate that it will be used in a shader as a place to store the computation result.

## Shader time

### Shader what?

A compute shader is a set of instructions that will be given to your GPU to tell it what calculations are needed.

In the same way that a CPU program can be written in multiple languages (rust, C, C++, ...), a GPU program can be written with another set of languages (GLSL, HLSL, SIR-V) that need to be compiled as well.

It could be a mess, but `wgpu` uses a universal shader translator, [ `naga` ](https://github.com/gfx-rs/naga) , that allow you to write your shader in `wgsl` or `glsl` , and make sure they are properly converted for each backend.

If you run your program on an Apple computer using the `metal` backend, your shader will be translated to the metal shading language (or `msl` ) automagically.

With all that being said, let's take a look at our `wgsl` instructions to convert an image from color to grayscale.

```wgsl
[[group(0), binding(0)]] var input_texture : texture_2d<f32>; 
[[group(0), binding(1)]] var output_texture : texture_storage_2d<rgba8unorm, write>; 

[[stage(compute), workgroup_size(16, 16)]]
fn main(
  [[builtin(global_invocation_id)]] global_id : vec3<u32>,
) {
    let coords = vec2<i32>(global_id.xy);
    let dimensions = textureDimensions(input_texture);

    if(coords.x >= dimensions.x || coords.y >= dimensions.y) {
        return;
    }

    let color = textureLoad(input_texture, coords.xy, 0);
    let gray = 0.299 * color.r + 0.587 * color.g + 0.114 * color.b;

    textureStore(output_texture, coords.xy, vec4<f32>(gray, gray, gray, color.a));
}
```

Contrarily to the CPU approach, where we would write one piece of code that iterates on every pixel to calculate it's grayscale value, the compute shader will be a piece of code that runs concurrently on each pixel.

We declare two variable, input and output texture. We can see here that the output is of the type `texture_storage_2d` , with the same `rgba8unorm` type as before.

Our main function declares a **workgroup size**, more on that later.

The rest is straightforward:
* Get the coordinates of the current pixel.
* Get the dimensions of the input image.
* If we are out of bounds, return.
* Load the pixel.
* Calculate the gray value of said pixel.
* Write it to the output texture.

> Having chosen the `Rbga8Unorm` format for our textures, the colors are retrieved as a float between 0 and 1, and we don't need to cast them when multiplying the r, g and b values to figure out the grayscale value.
>  
> If we had chosen instead the `Rbga8Uint` format instead, textureLoad would instead return a color of type `vec<u8>` , keeping the values between 0 and 255, and we would first need to cast them to float, before multipling them and recasting them to unsigned byte before writing down the output.

### Loading the shader and creating the pipeline

Okay, back to rust!

```rust
let shader = device.create_shader_module(&wgpu::ShaderModuleDescriptor {
    label: Some("Grayscale shader"),
    source: wgpu::ShaderSource::Wgsl(include_str!("shaders/grayscale.wgsl").into()),
});

let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
    label: Some("Grayscale pipeline"),
    layout: None,
    module: &shader,
    entry_point: "main",
});
```

Our shader is loaded as text, and we include it in our codebase for simplicity. We then proceed to create the compute pipeline.

## Bind group

We then proceed to creating our bind group: it is the rust representation of the data that will be attached to the gpu:
In the shader, we annotated our input_texture with `[[group(0), binding(0)]]` . We must now tell our rust code what this corresponds too.

```rust
let texture_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Texture bind group"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(
                    &input_texture.create_view(&wgpu::TextureViewDescriptor::default()),
                ),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(
                    &output_texture.create_view(&wgpu::TextureViewDescriptor::default()),
                ),
            },
        ],
    });
```

In this example, we bind two textures, but we could also bind data buffers or a texture sampler if we wanted. For the group 0, we match our `input_texture` to the binding 0, and our `output_texture` to the binding 1, just like in the shader!

> `pipeline.get_bind_group_layout(0)` automatically creates a bind group layout for us, based on the shader. Alternatively, we could create the bind group layout by hand instead, to be even more specific.

## Workgroup and dispatch

### Workgroup ?

Didn't I tell you that we would speak about workgroup?

According to [the wgsl spec](https://www.w3.org/TR/WGSL/#compute-shader-workgroups), "a workgroup is a set of invocations which concurrently execute a compute shader stage entry point, and share access to shader variables in the workgroup address space."

In our shader, we specified a workgroup of dimension 16 by 16. It can be seen as 2D matrix of instructions executed at once. In our case, 16 by 16 equals 256. Our shader will process when running 256 pixels at once! Take that, sequential computing!

Of course, our image is a bit bigger than 16x16, so we need to call this compute shader multiple times to handle every single pixels.

How many times exactly? Well, we simply divide the width and height of our image by the workgroup dimensions, and it will tell us how many times we need to run this 16x16 matrix to cover everything. We also make sure that to cover every pixels.

Let's have a simple helper method for that:

```rust
fn compute_work_group_count(
    (width, height): (u32, u32),
    (workgroup_width, workgroup_height): (u32, u32),
) -> (u32, u32) {
    let width = (width + workgroup_width - 1) / workgroup_width;
    let height = (height + workgroup_height - 1) / workgroup_height;

    (width, height)
}
```

This method makes sure that there will be enough workgroup to cover each pixels. 

> If we had a width of 20 pixels, and workgroup width of 16, we would be missing 4 pixels by only calling our shader once. We would need to call our shader 2 times, and it would then be able to cover 32 pixels in width. It's a bit of a waste, as some work will go to waste, but it's better than not applying our filters to some pixels.

### Dispatching

```rust
let mut encoder =
    device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
{
    let (dispatch_with, dispatch_height) =
        compute_work_group_count((texture_size.width, texture_size.height), (16, 16));
    let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("Grayscale pass"),
    });
    compute_pass.set_pipeline(&pipeline);
    compute_pass.set_bind_group(0, &texture_bind_group, &[]);
    compute_pass.dispatch(dispatch_with, dispatch_height, 1);
}

queue.submit(Some(encoder.finish()));
```

And now we create our compute pass, set our pipeline, bind our textures, and dispatch!

Dispatching tells `wgpu` how many invocation of the shader, or how many workgroups, must be created in each dimension. As you can see, there is a third argument, that we set to 1: workgroup can also be defined in three dimensions! But we are working on 2d textures, so we won't use it.

> For a picture of 32x32 pixels, we would need to dispatch 4 workgroups: 2 in the `x` dimensions, and 2 in the `y` dimensions.

### Global Invocation Id

So how do we go from workgroup to pixel position?
Simple: we used in the shader the `global_invocation_id` built-in variable! The `global_invocation_id` gives us the coordinate triple for the current invocation's corresponding compute shader grid point. Hum, I feel that is not helping so much. Let's just say that it multiplies the current workgroup identifier (our dispatch action create several workgroup, and gives to each of them a `x` and a `y` ) with the workgroup size, and add to it the `local_invocation_id` , meaning the coordinates of the current invocation within it's workgroup.

> Let's start again with our 32x32 image. 4 workgroup will be created, with ids (0, 0), (0, 1), (1, 0), (1, 1).
>  
> When the workgroup (1, 0) is running, 256 invocation will be running in parallel, with their own local identifier within the group: (0, 1), ... (0, 15), (1, 0) ... (7, 8) ... (15, 15).
>
> If we take the invocation (7, 8) of the workgroup (0, 1), it's global invocation id will be (0 * 16 + 7, 1 * 16 + 8), meaning (7, 24).
>  
> And that looks a lot like coordinates.

## Output

After all this work, you have it! The gray sushi!

![Gray sushi](sushi-grayscale.png)

# Final thoughts

So, is it worth it?

You tell me! For this example in particular, definitely not! Iterating over an array of pixels, running a O(n) algorithm to change the color to gray... a CPU will do such a great job there that it is not worth the hassle of all that code.

But it was a fun thing to do!

# A few links

First of, if you want to build it and run it yourself, you will find the code here:
* [code-sample/image-filters](https://github.com/redwarp/blog/tree/main/code-sample/image-filters)

I made a few other filters for fun, including a slightly more involved gaussian blur:
* [redwarp/filters](https://github.com/redwarp/filters)

Several useful links:
* [wgpu-rs](https://wgpu.rs/) - homepage to the `wgpu-rs` project.
* [Get started with GPU Compute on the web](https://web.dev/gpu-compute/#compute-shader-code) - it helped and inspired me to write this article.
* [The sushi picture, by gnokii](https://openclipart.org/detail/132169/sushi) - royalty free sushi.
* [WGSL Spec](https://www.w3.org/TR/WGSL) - read the doc, it helps!
