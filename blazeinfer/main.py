from .executor.model_executor import SimpleModelExecutor
from .generate import generate_text_naively
import logging


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')


def main():
    # Llama 3.1 is a great choice for conversational tasks.
    model_id = "meta-llama/Llama-3.1-8B-Instruct"

    try:
        # Step 1: Create the executor. This will load the model.
        executor = SimpleModelExecutor(model_id=model_id)

        # Step 2: Start a conversational loop
        while True:
            # Get input from the user
            prompt = input("\nEnter your prompt (or type 'exit' to quit): ")

            # Check for exit condition
            if prompt.strip().lower() == "exit":
                print("Exiting BlazeInfer. Goodbye!")
                break

            # Run the naive generation loop with the user's prompt
            print("\nBlazeInfer: ", end="", flush=True)
            generate_text_naively(executor, prompt, max_new_tokens=50)
            print("\n" + "="*50)
    except ImportError:
        logger.info("\nError: Please install transformers and torch.")
        logger.info("Run: pip install transformers torch")
    except Exception as e:
        logger.info(f"\nAn error occurred: {e}")
        logger.info("If using a gated model (like Llama),")
        logger.info("please ensure you are logged in: `huggingface-cli login`")

if __name__ == "__main__":
    logger.info("Starting BlazeInfer...")
    main()