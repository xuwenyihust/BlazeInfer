# Version 0

A minimalist LLM inference framework.

Clarity over performance, avoids complex optimizations like
- KV Cache
- Custom Kernels
- Quantization
- etc.

## Core Workflow: Naive Autoregressive Generation

The generation process follows a simple, step-by-step autoregressive loop. At each step, the model re-processes the **entire sequence** of known tokens to predict the next one. This is intentionally inefficient to clearly demonstrate the fundamental logic, as implemented in `generate_text_naively`.

1.  **Tokenization**:
    - The input `prompt` is converted into a sequence of token IDs by the `tokenizer`.

2.  **Autoregressive Loop**: For each new token to be generated:
    -   **Forward Pass**
        - The *entire* current sequence of token IDs (prompt + already generated tokens) is passed to the model. 
        - The model returns `logits`, which are raw scores for every word in the vocabulary at every position in the sequence.
            - `logits` Tensor
                - The model's raw, unnormalized predictions.
                - Returned by executor.forward()
                - Shape: [batch_size, sequence_length, vocab_size]
                - For every single token in your input sequence (sequence_length), the model outputs a giant list of scores—one score for every possible token in the entire vocabulary (vocab_size).

    -   **Select Next Token's Logits**
        - We only care about the prediction for the very next token. Therefore, we isolate the logits corresponding to the *last* token in the sequence.
            - `next_token_logits = logits[:, -1, :]`

    -   **Greedy Sampling**
        - To choose the next token, we simply select the one with the highest score (the `argmax`) from the next-token logits. This is the most straightforward sampling method.

    -   **Append**
        - The newly chosen token ID is appended to the end of our input sequence.

3.  **Repeat or Terminate**:
    - The loop continues until `max_new_tokens` is reached or the model generates an End-of-Sequence (`eos`) token.

4.  **Decode**:
    - The sequence of newly generated token IDs is decoded back into a human-readable string.