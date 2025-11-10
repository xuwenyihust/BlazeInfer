from .executor.model_executor import SimpleModelExecutor
from .generate import generate_text_naively
import logging


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')


def main():
    # Let's use a very small, fast model for this example
    # model_id = "gpt2"
    
    # Or, if you have access, a Llama model:
    # model_id = "meta-llama/Llama-3.1-8B-Instruct"
    
    model_id = "gpt2" # Using gpt2 as it's small and requires no login

    try:
        # Step 1: Create the executor. This will load the model.
        executor = SimpleModelExecutor(model_id=model_id)
        
        # Step 2: Run the naive generation loop
        prompt = "Hello, my name is"
        generate_text_naively(executor, prompt, max_new_tokens=50)

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