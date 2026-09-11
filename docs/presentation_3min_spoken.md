# 3-minute talk, general audience, no slides

Read at a normal pace this runs about 3 minutes.  The four marks are where to pause.

---

Imagine a satellite or a laser scanner passing over a city.  What it gives you is not a model - it
is a cloud of measured points floating in space, a million of them, with nothing connecting them.

Before you can do anything useful - measure a volume, simulate water flowing over the terrain, build
a 3D model - you have to connect those points into solid pieces.  The standard way is to fill the
space with tetrahedra: little four-cornered pyramids, the 3D version of covering a surface with
triangles.  There is one particular way of doing it, called the Delaunay triangulation, that is
mathematically the best behaved, and it is what essentially every tool uses.  A million measured
points turns into about seven million tetrahedra.

*(pause)*

Doing that takes seconds on a normal processor, which is slow when it sits inside a bigger pipeline.
So for more than ten years, researchers have been trying to do it on graphics cards instead - the
same chips that render video games - because they can do thousands of things at once.  Several
papers report being five or ten times faster.

Here is what caught our attention.  All of those results were measured on **artificial** data:
points scattered completely at random.  Real measured data is not random at all.  A sensor samples
on a regular grid, and that regularity creates situations where the geometry is ambiguous - where
there is no single right answer, and the software has to make a delicate decision.  Nobody had
tested the fast methods on that kind of data.

*(pause)*

So we did.  We took the reference software that the field trusts - slow, but exact - and we took
three published graphics-card methods, and we ran all of them on the same real point clouds, on the
same machine, and compared the results piece by piece: not just how long they took, but whether they
produced the same mesh.

Getting there was half the work.  Two of the three programs no longer even compiled - one of them
relies on a feature that graphics cards removed years ago - so we had to repair them first.

*(pause)*

And the results were not what the papers led us to expect.  On real data, one method quietly threw
away a third of the mesh and reported success.  Another crashed or froze on nine out of ten of our
tests.  A third slowed down by a factor of five for no visible reason.  The fastest method that was
also always correct turned out not to be a graphics card method at all - it was a program running on
an ordinary processor.  We also found that the reference software's own "fast parallel mode" is
slower than its plain single-threaded mode on this data, which means the comparison everyone quotes
is against the wrong baseline.

The lesson is not that graphics cards do not work.  It is that the speed-ups were real, but measured
on data that does not look like the real world - and on real data, being correct and not crashing
matters more than being fast.
