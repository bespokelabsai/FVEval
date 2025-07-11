# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict, List, Optional, Any, Union
import os
import pandas as pd
from datasets import Dataset
from bespokelabs import curator
from dataclasses import asdict
import logging

from fv_eval import (
    prompts_design2sva,
    prompts_nl2sva_machine,
    prompts_nl2sva_human,
    utils,
)
from fv_eval.data import InputData, LMResult

"""
Curator-based Benchmark Launcher for FVEval
Uses Bespoke Curator for efficient LLM inference on datasets
"""

# Set up logging for Curator to see batch processing details
logger = logging.getLogger("bespokelabs.curator")
logger.setLevel(logging.INFO)


class CuratorBenchmarkLauncher:
    """Base class for Curator-based benchmark launching."""
    
    def __init__(
        self,
        save_dir: str,
        dataset_path: str,
        task: str,
        model_name_list: List[str],
        debug: bool = False,
    ):
        self.save_dir = save_dir
        self.dataset_path = dataset_path
        self.task = task
        self.model_name_list = model_name_list
        self.debug = debug
        
        # Load dataset as HuggingFace Dataset (preferred by Curator)
        df = pd.read_csv(dataset_path)
        if self.debug:
            # Only take first 2 rows for debugging
            df = df.head(2)
            
        # Convert to Dataset and ensure all columns are strings to avoid type issues
        self.dataset = Dataset.from_pandas(df)
        self.experiment_id = dataset_path.split(".csv")[0].split("/")[-1]
        
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
    
    def _get_curator_backend_config(self, model_name: str) -> tuple[str, str, Dict[str, Any]]:
        """Convert model name to Curator backend configuration."""
        backend_params = {}
        original_model_name = model_name
        
        if "vllm" in model_name:
            backend = "openai"  # Use OpenAI-compatible backend for vLLM
            backend_params = {
                "base_url": "http://localhost:8000/v1",
                "api_key": "EMPTY"
            }
            # For vLLM, we need to get the actual model name from the server
            try:
                from openai import OpenAI
                client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")
                model_name = client.models.list().data[0].id
            except Exception as e:
                utils.print_error("ERROR", f"Failed to get vLLM model name: {e}")
                raise
            
        elif model_name in [
            "llama-3-8b", "llama-3-70b", "llama-3.1-8b", "llama-3.1-70b",
            "gemma-2-27b", "dbrx", "qwen-2-72b", "llama-2-70b",
            "mixtral-8x22b", "mixtral-8x7b"
        ]:
            backend = "openai"  # Use Together via OpenAI-compatible interface
            together_model_dict = {
                "llama-3-8b": "meta-llama/Llama-3-8b-chat-hf",
                "llama-3-70b": "meta-llama/Llama-3-70b-chat-hf", 
                "llama-3.1-8b": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
                "llama-3.1-70b": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
                "gemma-2-27b": "google/gemma-2-27b-it",
                "dbrx": "databricks/dbrx-instruct",
                "qwen-2-72b": "Qwen/Qwen2-72B-Instruct",
                "llama-2-70b": "meta-llama/Llama-2-70b-chat-hf",
                "mixtral-8x22b": "mistralai/Mixtral-8x22B-Instruct-v0.1",
                "mixtral-8x7b": "mistralai/Mixtral-8x7B-Instruct-v0.1",
            }
            model_name = together_model_dict[model_name]
            backend_params = {
                "base_url": "https://api.together.xyz/v1",
                "api_key": os.getenv("TOGETHER_API_KEY"),
                "max_requests_per_minute": 100,
                "max_tokens_per_minute": 50_000,
            }
            
        elif "claude" in model_name:
            backend = "anthropic"
            if "3.5" in model_name:
                model_name = "claude-3-5-sonnet-20240620"
            elif "opus" in model_name:
                model_name = "claude-3-opus-20240229"
            elif "sonnet" in model_name:
                model_name = "claude-3-sonnet-20240229"
            elif "haiku" in model_name:
                model_name = "claude-3-haiku-20240307"
            else:
                raise ValueError(f"Unknown Anthropic model: {model_name}")
            backend_params = {
                "max_requests_per_minute": 50,
                "max_tokens_per_minute": 40_000,
            }
            
        elif "gemini" in model_name:
            backend = "litellm"
            if "flash" in model_name:
                model_name = "gemini/gemini-1.5-flash"
            else:
                model_name = "gemini/gemini-1.5-pro"
            backend_params = {
                "max_requests_per_minute": 100,
                "max_tokens_per_minute": 50_000,
            }
            
        elif "gpt" in model_name:
            backend = "openai"
            # if "gpt-4-turbo" in model_name:
            #     model_name = "gpt-4-0125-preview"
            # elif model_name == "gpt-4o":
            #     model_name = "gpt-4o-2024-05-13"
            # elif model_name == "gpt-4":
            #     model_name = "gpt-4-0613"
            # elif "gpt-3.5-turbo" in model_name:
            #     model_name = "gpt-3.5-turbo-0125"
            # backend_params = {
            #     "max_requests_per_minute": 500,
            #     "max_tokens_per_minute": 150_000,
            # }
        else:
            raise ValueError(f"Unknown model: {model_name}")
            
        return backend, model_name, backend_params
    
    def create_generator_class(self):
        """Override in subclasses to return the appropriate generator class."""
        raise NotImplementedError
    
    def run_benchmark(
        self,
        temperature: float = 0.0,
        max_tokens: int = 100,
        cot_strategy: str = "default",
        num_cases: int = 1,
    ):
        """Run benchmark using Curator generators."""
        results_list = []
        
        for short_model_name in self.model_name_list:
            backend, model_name, backend_params = self._get_curator_backend_config(short_model_name)
            
            # Create generator instance
            generator_class = self.create_generator_class()
            generator = generator_class(
                model_name=model_name,
                backend=backend,
                backend_params=backend_params,
                generation_params={
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "top_p": 1.0 if temperature == 0.0 else 0.95,
                },
                experiment_id=self.experiment_id,
                cot_strategy=cot_strategy,
                num_cases=num_cases,
            )
            
            # Process dataset with Curator
            if self.debug:
                print(f"Processing {len(self.dataset)} samples with {short_model_name}")
                print(f"Model: {model_name}, Backend: {backend}")
            
            try:
                results = generator(self.dataset)
                self.save_results(short_model_name, results)
                results_list.append(results)
                
                if self.debug:
                    # Get length from the appropriate source
                    if hasattr(results, 'dataset'):
                        result_count = len(results.dataset)
                    elif hasattr(results, '__len__'):
                        result_count = len(results)
                    else:
                        result_count = "unknown"
                    print(f"Successfully processed {result_count} results")
                    
            except Exception as e:
                utils.print_error("ERROR", f"Failed to process {short_model_name}: {e}")
                if self.debug:
                    raise
                
        return results_list
    
    def save_results(self, model_name: str, curator_result):
        """Save Curator results to CSV."""
        model_name = model_name.split("/")[-1].replace(" ", "_")
        
        # Handle new CuratorResponse format
        if hasattr(curator_result, 'dataset'):
            # New CuratorResponse object
            results_df = curator_result.dataset.to_pandas()
        elif hasattr(curator_result, 'to_pandas'):
            # Legacy Dataset object
            results_df = curator_result.to_pandas()
        else:
            # Fallback - try to access as Dataset directly
            results_df = curator_result.to_pandas()
        
        # Ensure we have all required columns for compatibility
        required_columns = ["experiment_id", "task_id", "model_name", "response", 
                          "ref_solution", "user_prompt", "output_tb", "design_rtl", "cot_response"]
        for col in required_columns:
            if col not in results_df.columns:
                results_df[col] = ""
        
        output_path = os.path.join(self.save_dir, f"{model_name}_{self.experiment_id}.csv")
        results_df.to_csv(output_path, index=False)
        
        if self.debug:
            print(f"Saved results to: {output_path}")
            print(f"Results shape: {results_df.shape}")
            if len(results_df) > 0:
                print(f"Sample columns: {list(results_df.columns)}")


class NL2SVAHumanCuratorGenerator(curator.LLM):
    """Curator-based generator for NL2SVA Human task."""
    
    def __init__(
        self,
        num_icl_examples: int = 3,
        use_cot: bool = False,
        experiment_id: str = "",
        cot_strategy: str = "default",  # For compatibility
        num_cases: int = 1,  # For compatibility
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_icl_examples = num_icl_examples
        self.use_cot = use_cot
        self.experiment_id = experiment_id
        
        # Store for use in parse method
        self._current_model_name = kwargs.get('model_name', 'unknown')
    
    def prompt(self, input: dict) -> str:
        """Generate the complete prompt for the model."""
        # Generate user prompt prefix (ICL examples + testbench)
        if self.use_cot:
            user_prompt_prefix = prompts_nl2sva_human.SVAGEN_HUMAN_ICL_EXAMPLE_3_COT
        elif self.num_icl_examples == 0:
            user_prompt_prefix = ""
        elif self.num_icl_examples == 1:
            user_prompt_prefix = prompts_nl2sva_human.SVAGEN_HUMAN_ICL_EXAMPLE_1
        elif self.num_icl_examples == 3:
            user_prompt_prefix = prompts_nl2sva_human.SVAGEN_HUMAN_ICL_EXAMPLE_3
        else:
            raise ValueError(f"Unsupported number of in-context examples: {self.num_icl_examples}")
        
        user_prompt_prefix += "\n\n" + prompts_nl2sva_human.SVAGEN_TB_PREAMBLE
        user_prompt_prefix += "\n" + input["testbench"]
        
        # Generate question prompt
        question_prompt = prompts_nl2sva_human.SVAGEN_QUESTION_PREAMBLE
        question_prompt += input["prompt"] + "\n"
        question_prompt += (
            prompts_nl2sva_human.SVAGEN_QUESTION_POSTAMBLE_COT if self.use_cot 
            else prompts_nl2sva_human.SVAGEN_QUESTION_POSTAMBLE
        )
        
        # For models that support system prompts, we need to handle them properly
        full_prompt = user_prompt_prefix + "\n" + question_prompt
        
        # Add system prompt as part of the user message for models that don't separate them
        if hasattr(self, 'backend') and self.backend in ['openai', 'anthropic']:
            # These backends handle system prompts separately
            return full_prompt
        else:
            # Include system prompt in the user message
            system_prompt = prompts_nl2sva_human.SVAGEN_HEADER
            return f"{system_prompt}\n\n{full_prompt}"
    
    def parse(self, input: dict, response: str) -> dict:
        """Parse response and package testbench."""
        # Generate question prompt for comments
        question_prompt = prompts_nl2sva_human.SVAGEN_QUESTION_PREAMBLE + input["prompt"] + "\n"
        
        # Package testbench
        reference_assertion_text = input["ref_solution"].replace("asrt", "reference")
        assertion_text = utils.parse_code_response(response)
        
        commented_question_text = "\n//".join(question_prompt.split("\n"))
        testbench_text = input["testbench"]
        packaged_tb_text = (
            testbench_text.split("endmodule")[0]
            + "\n\n"
            + commented_question_text
            + "\n\n"
            + reference_assertion_text
            + "\n\n"
            + assertion_text
            + "\n\n"
            + "endmodule"
        )
        
        return {
            "experiment_id": self.experiment_id,
            "task_id": input["design_name"] + "_" + input["task_id"] + "_trial_0",
            "model_name": self._current_model_name,
            "response": response,
            "ref_solution": input["ref_solution"],
            "user_prompt": self.prompt(input),
            "output_tb": packaged_tb_text,
            "design_rtl": "\n",
            "cot_response": "cot_response\n",
        }


class NL2SVAMachineCuratorGenerator(curator.LLM):
    """Curator-based generator for NL2SVA Machine task."""
    
    def __init__(
        self,
        num_icl_examples: int = 3,
        use_cot: bool = False,
        experiment_id: str = "",
        cot_strategy: str = "default",  # For compatibility
        num_cases: int = 1,  # For compatibility
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_icl_examples = num_icl_examples
        self.use_cot = use_cot
        self.experiment_id = experiment_id
        
        # Store for use in parse method
        self._current_model_name = kwargs.get('model_name', 'unknown')
    
    def prompt(self, input: dict) -> str:
        """Generate the complete prompt for the model."""
        # Generate user prompt prefix (ICL examples only, no testbench)
        if self.use_cot:
            user_prompt_prefix = prompts_nl2sva_machine.SVAGEN_MACHINE_ICL_EXAMPLE_3_COT
        elif self.num_icl_examples == 0:
            user_prompt_prefix = ""
        elif self.num_icl_examples == 1:
            user_prompt_prefix = prompts_nl2sva_machine.SVAGEN_MACHINE_ICL_EXAMPLE_1
        elif self.num_icl_examples == 2:
            user_prompt_prefix = prompts_nl2sva_machine.SVAGEN_MACHINE_ICL_EXAMPLE_2
        elif self.num_icl_examples == 3:
            user_prompt_prefix = prompts_nl2sva_machine.SVAGEN_MACHINE_ICL_EXAMPLE_3
        else:
            raise ValueError(f"Unsupported number of in-context examples: {self.num_icl_examples}")
        
        # Generate question prompt
        question_prompt = prompts_nl2sva_machine.SVAGEN_QUESTION_PREAMBLE + input["prompt"] + "\n"
        
        if self.use_cot:
            question_prompt += prompts_nl2sva_machine.SVAGEN_QUESTION_POSTAMBLE_COT
        elif self.num_icl_examples == 0:
            question_prompt += prompts_nl2sva_machine.SVAGEN_QUESTION_POSTAMBLE_ZERO_SHOT
        else:
            question_prompt += prompts_nl2sva_machine.SVAGEN_QUESTION_POSTAMBLE
        
        full_prompt = user_prompt_prefix + "\n" + question_prompt
        
        # Handle system prompt
        if hasattr(self, 'backend') and self.backend in ['openai', 'anthropic']:
            return full_prompt
        else:
            system_prompt = prompts_nl2sva_machine.SVAGEN_HEADER
            return f"{system_prompt}\n\n{full_prompt}"
    
    def parse(self, input: dict, response: str) -> dict:
        """Parse response and package testbench."""
        # Similar to human version but without testbench context
        question_prompt = prompts_nl2sva_machine.SVAGEN_QUESTION_PREAMBLE + input["prompt"] + "\n"
        
        reference_assertion_text = input["ref_solution"].replace("asrt", "reference")
        assertion_text = utils.parse_code_response(response)
        
        commented_question_text = "\n//".join(question_prompt.split("\n"))
        testbench_text = input["testbench"]
        packaged_tb_text = (
            testbench_text.split("endmodule")[0]
            + "\n\n"
            + commented_question_text
            + "\n\n"
            + reference_assertion_text
            + "\n\n"
            + assertion_text
            + "\n\n"
            + "endmodule"
        )
        
        return {
            "experiment_id": self.experiment_id,
            "task_id": input["design_name"] + "_" + input["task_id"] + "_trial_0",
            "model_name": self._current_model_name,
            "response": response,
            "ref_solution": input["ref_solution"],
            "user_prompt": self.prompt(input),
            "output_tb": packaged_tb_text,
            "design_rtl": "\n",
            "cot_response": "cot_response\n",
        }


class Design2SVACuratorGenerator(curator.LLM):
    """Curator-based generator for Design2SVA task."""
    
    def __init__(
        self,
        cot_strategy: str = "default",
        num_cases: int = 1,
        experiment_id: str = "",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.cot_strategy = cot_strategy
        self.num_cases = num_cases
        self.experiment_id = experiment_id
        
        # Store for use in parse method
        self._current_model_name = kwargs.get('model_name', 'unknown')
        
        # Get CoT strategy - for now, we'll use the direct approach
        # Complex CoT would require sequential calls
        self.cot_chain = self._get_cot_strategy(cot_strategy)
    
    def _get_cot_strategy(self, cot_strategy: str) -> List[tuple[str, str]]:
        """Get chain of thought strategy."""
        if cot_strategy == "default":
            return [
                ("question", prompts_design2sva.get_design2sva_direct_question_prompt(1))
            ]
        elif cot_strategy == "plan-act":
            # For now, simplify to direct approach
            # TODO: Implement proper CoT with sequential calls
            utils.print_error("WARNING", "CoT strategies not fully implemented in Curator version, using default")
            return [
                ("question", prompts_design2sva.get_design2sva_direct_question_prompt(1))
            ]
        elif cot_strategy == "plan-model-act":
            # For now, simplify to direct approach
            utils.print_error("WARNING", "CoT strategies not fully implemented in Curator version, using default")
            return [
                ("question", prompts_design2sva.get_design2sva_direct_question_prompt(1))
            ]
        else:
            raise ValueError(f"Unsupported COT strategy: {cot_strategy}")
    
    def prompt(self, input: dict) -> str:
        """Generate the complete prompt for the model."""
        # Generate user prompt prefix
        testbench_text = input["testbench"]
        testbench_text = testbench_text.split("assign tb_reset")[0]
        testbench_text += "assign tb_reset = (reset_ == 1'b0);\n"
        
        user_prompt_prefix = prompts_design2sva.SVAGEN_DUT_PREAMBLE
        user_prompt_prefix += input["prompt"]  # This is the design RTL
        user_prompt_prefix += "\n\n" + prompts_design2sva.SVAGEN_TB_PREAMBLE
        user_prompt_prefix += "\n" + testbench_text
        
        # Use the direct question approach
        question = prompts_design2sva.get_design2sva_direct_question_prompt(1)
        
        full_prompt = user_prompt_prefix + "\n" + question
        
        # Handle system prompt
        if hasattr(self, 'backend') and self.backend in ['openai', 'anthropic']:
            return full_prompt
        else:
            system_prompt = prompts_design2sva.SVAGEN_HEADER
            return f"{system_prompt}\n\n{full_prompt}"
    
    def parse(self, input: dict, response: str) -> dict:
        """Parse response and package testbench."""
        # Package testbench
        testbench_text_prefix = input["testbench"]
        testbench_text_prefix = testbench_text_prefix.split("assign tb_reset")[0]
        testbench_text_prefix += "assign tb_reset = (reset_ == 1'b0);\n"
        testbench_text_postfix = "endmodule\n" + input["testbench"].split("endmodule")[-1]
        
        lm_response = utils.parse_code_response(response)
        packaged_tb_text = (
            testbench_text_prefix + "\n" + lm_response + "\n" + testbench_text_postfix
        )
        
        return {
            "experiment_id": self.experiment_id,
            "task_id": input["design_name"] + "_" + input["task_id"] + "_trial_0",
            "model_name": self._current_model_name,
            "response": response,
            "ref_solution": input["ref_solution"],
            "user_prompt": self.prompt(input),
            "output_tb": packaged_tb_text,
            "design_rtl": input["prompt"],  # Design RTL is in the prompt field
            "cot_response": "cot_response\n",
        }


# Launcher classes that use the Curator generators
class NL2SVAHumanLauncher(CuratorBenchmarkLauncher):
    """Launcher for NL2SVA Human task using Curator."""
    
    def __init__(
        self,
        save_dir: str,
        dataset_path: str,
        task: str = "nl2sva_human",
        model_name_list: List[str] = ["gpt-4"],
        num_icl_examples: int = 3,
        use_cot: bool = False,
        debug: bool = False,
    ):
        super().__init__(save_dir, dataset_path, task, model_name_list, debug)
        self.num_icl_examples = num_icl_examples
        self.use_cot = use_cot
    
    def create_generator_class(self):
        def create_generator(**kwargs):
            return NL2SVAHumanCuratorGenerator(
                num_icl_examples=self.num_icl_examples,
                use_cot=self.use_cot,
                **kwargs
            )
        return create_generator


class NL2SVAMachineLauncher(CuratorBenchmarkLauncher):
    """Launcher for NL2SVA Machine task using Curator."""
    
    def __init__(
        self,
        save_dir: str,
        dataset_path: str,
        task: str = "nl2sva_machine",
        model_name_list: List[str] = ["gpt-4"],
        num_icl_examples: int = 3,
        use_cot: bool = False,
        debug: bool = False,
    ):
        super().__init__(save_dir, dataset_path, task, model_name_list, debug)
        self.num_icl_examples = num_icl_examples
        self.use_cot = use_cot
    
    def create_generator_class(self):
        def create_generator(**kwargs):
            return NL2SVAMachineCuratorGenerator(
                num_icl_examples=self.num_icl_examples,
                use_cot=self.use_cot,
                **kwargs
            )
        return create_generator


class Design2SVALauncher(CuratorBenchmarkLauncher):
    """Launcher for Design2SVA task using Curator."""
    
    def __init__(
        self,
        save_dir: str,
        dataset_path: str,
        task: str = "design2sva",
        model_name_list: List[str] = ["gpt-4"],
        use_cot: bool = False,
        debug: bool = False,
    ):
        super().__init__(save_dir, dataset_path, task, model_name_list, debug)
        self.use_cot = use_cot
    
    def create_generator_class(self):
        def create_generator(**kwargs):
            return Design2SVACuratorGenerator(**kwargs)
        return create_generator


# For backward compatibility with existing code
BenchmarkLauncher = CuratorBenchmarkLauncher
NL2SVALauncher = CuratorBenchmarkLauncher