from PIL import Image, ImageOps, ImageDraw, ImageFilter
import os


def add_canvas_frame(
    input_path="img.png",
    output_path="framed_artwork.jpg",
    canvas_size=60,
    inner_line_margin=10,
    shadow_offset=12,
    background_color=(245, 245, 240),
):
    """
    Add a clean white canvas-style border to an image and save as JPG.

    Parameters:
        input_path (str): Input PNG image path.
        output_path (str): Output JPG image path.
        canvas_size (int): Thickness of the white border.
        inner_line_margin (int): Margin for subtle inner outline.
        shadow_offset (int): Drop shadow offset.
        background_color (tuple): Background color behind the artwork.
    """

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Load image
    img = Image.open(input_path).convert("RGB")

    # Add white canvas border
    framed = ImageOps.expand(img, border=canvas_size, fill=(250, 250, 247))
    fw, fh = framed.size

    # Add subtle inner outline for a mounted-art look
    draw = ImageDraw.Draw(framed)
    m = inner_line_margin
    draw.rectangle(
        [m, m, fw - m - 1, fh - m - 1],
        outline=(220, 220, 215),
        width=2,
    )

    # Prepare background canvas
    pad = 10
    canvas_w = fw + shadow_offset * 2 + pad * 2
    canvas_h = fh + shadow_offset * 2 + pad * 2
    background = Image.new("RGB", (canvas_w, canvas_h), background_color)

    # Create shadow
    shadow = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow)

    pos_x = pad
    pos_y = pad

    shadow_draw.rectangle(
        [
            pos_x + shadow_offset,
            pos_y + shadow_offset,
            pos_x + fw + shadow_offset,
            pos_y + fh + shadow_offset,
        ],
        fill=(0, 0, 0, 80),
    )
    shadow = shadow.filter(ImageFilter.GaussianBlur(14))

    # Composite all layers
    result = background.convert("RGBA")
    result.alpha_composite(shadow)
    result.alpha_composite(framed.convert("RGBA"), (pos_x, pos_y))

    # Save as JPG
    result = result.convert("RGB")
    result.save(output_path, "JPEG", quality=95)

    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    add_canvas_frame(
        input_path="figures\\felt\\stage2_felt_mallow.png",
        output_path="social_media_images\\felt_mallow.jpg",
        canvas_size=26,
        inner_line_margin=8,
        shadow_offset=3,
    )