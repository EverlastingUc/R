# Rumelhart, Hinton, and Williams (1986)

## Citation

David E. Rumelhart, Geoffrey E. Hinton, and Ronald J. Williams. “Learning
representations by back-propagating errors.” *Nature*, 323, 533–536 (1986).
[Nature publisher page](https://doi.org/10.1038/323533a0).

## Analysis

- **Established — method:** The authors describe backpropagation as repeatedly
  adjusting connection weights to reduce the difference between a network's
  actual output and its desired output.
- **Established — hidden units:** The publisher abstract explains that hidden
  units can come to represent useful features of the task as the weights
  change.
- **Inference — why it matters to Entropy:** This is the clearest conceptual
  link to our training loop. Entropy calculates output error, sends its effect
  backward through the layers, and adjusts weights so the next predictions
  improve.
- **Inference — limitation for this project:** The paper describes a learning
  method; it does not imply that every network trained this way will generalize
  or that a small XOR model can perform language tasks. Those claims require
  separate measurements.

## Evidence and access

The title, authors, date, journal, pages, and method summary were checked on
the publisher page. The publisher abstract was accessible; the full article
was not available through the publisher page during this review, so the
analysis is limited to the abstract and bibliographic record.
