# Hopper Studio Paper Options

Use this as the Paper exploration brief before further frontend refinement in production code.

## Goal

Create three Paper frames for the new Hopper Studio site:

1. Homepage / generator
2. Result / publish step
3. Artwork detail / permalink

The UI should feel like a quiet gallery or small museum, not a generic AI tool.

## Shared Constraints

- Text-only prompt input
- One primary action on the homepage
- No visible size or advanced controls
- Museum-like typography
- Quiet palette
- Restrained motion
- Mobile-first clarity
- Reduced-motion safe

## Option A: Moving Background Wall

Direction:
The homepage feels like a softly animated salon wall, with published works drifting behind the prompt composer.

Frame 1, homepage / generator:
- Large centered prompt composer
- One oversized Generate button
- Published works as a slow drifting field behind translucent panels
- Copy feels editorial and sparse

Frame 2, result / publish:
- Freshly generated image fills most of the viewport inside a dark wood frame
- Right-side plaque area with prompt summary, Generate again, and Publish your artwork
- Publish form is calm and secondary to the image

Frame 3, artwork detail:
- Full framed artwork
- Plaque with title, attribution, and date
- Shareable permalink treatment under the plaque

Why choose it:
- Most atmospheric
- Best fit if the homepage should feel alive without adding controls

Risk:
- Motion can feel too decorative if the background wall is too busy

## Option B: Featured Carousel

Direction:
The homepage feels more curated and legible, like a museum site highlighting one featured published work at a time.

Frame 1, homepage / generator:
- Prompt composer on the left
- A single featured published work or small carousel on the right
- Stronger editorial hierarchy and less ambient motion

Frame 2, result / publish:
- Split layout with framed image and a clean publishing panel
- More visible structure around title and attribution inputs

Frame 3, artwork detail:
- Classic artwork page with centered framed image and formal metadata below

Why choose it:
- Clearest and easiest to ship
- Better if gallery readability matters more than atmosphere

Risk:
- Can drift toward a conventional marketing layout if not art-directed carefully

## Option C: Reading Room

Direction:
The site feels like a printed exhibition catalog translated into a web studio.

Frame 1, homepage / generator:
- Tall text area that reads like a writing desk
- Narrow column layout
- Published works appear as small archival thumbnails below the fold

Frame 2, result / publish:
- Image on top, publishing details below like catalog fields
- Strong typography and negative space instead of motion

Frame 3, artwork detail:
- Artwork with a catalog entry feel
- Date, attribution, and prompt shown as exhibition metadata

Why choose it:
- Most distinctive
- Strong fit for “quiet” and “museum-like”

Risk:
- Less immediately dynamic than the moving wall or carousel directions

## Recommended Starting Point

Start Paper with:

- Option A for the homepage
- Option B for the result / publish screen
- Option C typography cues for the detail page

This mix keeps the homepage atmospheric, the publish flow practical, and the permalink page more archival.

## Paper Prompt

Design three mobile-and-desktop web frames for "Hopper Studio", a text-only AI image generator inspired by Edward Hopper paintings. The product should feel like a quiet gallery or small museum, not a generic AI tool. Use museum-like typography, a warm paper palette, restrained motion, framed artwork presentation, and minimal controls. Frame 1 is the homepage with one large prompt field and one Generate button, plus published artworks shown either as a subtle moving wall or a curated carousel. Frame 2 is the result and publish step, showing the generated image in a frame with Generate again and Publish your artwork, plus only title and attribution inputs. Frame 3 is a public artwork permalink page with title plaque, attribution, date, and a shareable presentation. Keep the interface sparse, elegant, and calm, and ensure the mobile layout remains prompt-first and simple.

