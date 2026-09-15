import os
import math
from tqdm import tqdm
import cv2
import torch
import pandas as pd
import numpy as np
from PIL import Image
from typing import List, Dict, Union
import scipy.io.wavfile
from torch.utils.data import Dataset, DataLoader

from transformers import (
    Blip2Processor, 
    Blip2ForConditionalGeneration,
    CLIPProcessor, 
    CLIPModel,
    MusicgenForConditionalGeneration,
    AutoProcessor,
    AutoModelForCausalLM,
    AutoTokenizer
)

# 13 emotional categories from Cowen et al.
COWEN_13_EMOTIONS = [
    "amusing",
    "annoying",
    "anxious/tense",
    "beautiful",
    "calm/relaxing/serene",
    "dreamy",
    "energizing/pump-up",
    "erotic/desirous",
    "indignant/defiant",
    "joyful/cheerful",
    "sad/depressing",
    "scary/fearful",
    "triumphant/heroic"
]

class VideosDataset(Dataset):
    def __init__(self, data_pairs):
        self.data_pairs = data_pairs

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        # (video_input_path, custom_save_path)
        return self.data_pairs[idx]


class Herrmann1Pipeline:
    """
    Herrmann-1 Model Implementation (Vision + Emotion + LLM Prompting + MusicGen).
    Excludes speech/audio transcription components.
    Supports batched inference via DataLoader with custom output filename mapping.
    """
    def __init__(
        self,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype: torch.dtype = torch.float16,
        blip2_model_id: str = "Salesforce/blip2-opt-2.7b",
        clip_model_id: str = "openai/clip-vit-base-patch32",
        musicgen_model_id: str = "facebook/musicgen-medium",
        use_openai: bool = False,
        openai_api_key: str = None,
        llm_model_id: str = "Qwen/Qwen2.5-7B-Instruct" 
    ):
        self.device = device
        self.dtype = torch_dtype
        self.use_openai = use_openai

        print("--> Loading BLIP-2 model for video frame captioning...")
        self.blip_processor = Blip2Processor.from_pretrained(blip2_model_id)
        self.blip_model = Blip2ForConditionalGeneration.from_pretrained(
            blip2_model_id, torch_dtype=self.dtype, use_safetensors=True
        ).to(self.device)

        print("--> Loading CLIP model for zero-shot emotion analysis...")
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_id)
        self.clip_model = CLIPModel.from_pretrained(clip_model_id, use_safetensors=True).to(self.device)

        print("--> Loading MusicGen Medium model...")
        self.musicgen_processor = AutoProcessor.from_pretrained(musicgen_model_id)
        self.musicgen_model = MusicgenForConditionalGeneration.from_pretrained(
            musicgen_model_id, torch_dtype=self.dtype, use_safetensors=True
        ).to(self.device)

        # Initialize LLM
        if self.use_openai:
            import openai
            self.client = openai.OpenAI(api_key=openai_api_key)
        else:
            print(f"--> Loading Open-Source LLM ({llm_model_id})...")
            self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_id, padding_side="left")
            if self.llm_tokenizer.pad_token is None:
                self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                llm_model_id, torch_dtype=self.dtype, device_map="auto"
            )

    @staticmethod
    def sample_frames_from_video(video_path: str, num_frames: int = 15) -> List[Image.Image]:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames <= 0:
            raise ValueError(f"Could not read video or video is empty: {video_path}")

        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        frames = []

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame_rgb))
            else:
                break

        cap.release()
        return frames

    @torch.no_grad()
    def generate_captions_batch(self, images: List[Image.Image], batch_size: int = 8) -> List[str]:
        captions = []
        for i in range(0, len(images), batch_size):
            batch_imgs = images[i : i + batch_size]
            inputs = self.blip_processor(images=batch_imgs, return_tensors="pt").to(self.device, self.dtype)
            generated_ids = self.blip_model.generate(**inputs, max_new_tokens=40)
            batch_captions = self.blip_processor.batch_decode(generated_ids, skip_special_tokens=True)
            captions.extend([cap.strip() for cap in batch_captions])
        return captions

    @torch.no_grad()
    def analyze_emotions_batch(self, images: List[Image.Image], batch_size: int = 8) -> Dict[str, float]:
        prompt_templates = [f"a video scene that feels {emotion}" for emotion in COWEN_13_EMOTIONS]
        all_probs = []
        for i in range(0, len(images), batch_size):
            batch_imgs = images[i : i + batch_size]
            inputs = self.clip_processor(
                text=prompt_templates,
                images=batch_imgs,
                return_tensors="pt",
                padding=True
            ).to(self.device)

            outputs = self.clip_model(**inputs)
            logits_per_image = outputs.logits_per_image 
            probs = torch.softmax(logits_per_image, dim=-1)
            all_probs.append(probs.cpu())

        mean_probs = torch.cat(all_probs, dim=0).mean(dim=0).numpy()
        return {emotion: float(prob) for emotion, prob in zip(COWEN_13_EMOTIONS, mean_probs)}

    def construct_llm_prompt(self, captions: List[str], emotion_scores: Dict[str, float]) -> str:
        caption_str = " ".join([f"{i+1}) {cap}" for i, cap in enumerate(captions)])
        top_emotions = sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        sentiment_str = ", ".join([f"{emo} {score*100:.2f}%" for emo, score in top_emotions])

        prompt = f"""Given the following image captions from a video: {caption_str}
and given that the sentiments of the video are: {sentiment_str}
Describe the music that would fit such a video. Your output will be fed to a text to music model. To help you out, here are some prompts that worked well with the model: 
1) Pop dance track with catchy melodies, tropical percussion, and upbeat rhythms, perfect for the beach 
2) classic reggae track with an electronic guitar solo 
3) earthy tones, environmentally conscious, ukulele-infused, harmonic, breezy, easygoing, organic instrumentation, gentle grooves 
4) lofi slow bpm electro chill with organic samples 
5) violins and synths that inspire awe at the finiteness of life and the universe 
6) 80s electronic track with melodic synthesizers, catchy beat and groovy bass 

Give me only the description of the music without any explanation. Give me a single description."""
        return prompt

    def query_llm_batch(self, prompts: List[str]) -> List[str]:
        descriptions = []
        if self.use_openai:
            for prompt in prompts:
                response = self.client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_tokens=100
                )
                descriptions.append(response.choices[0].message.content.strip())
        else:
            inputs = self.llm_tokenizer(prompts, return_tensors="pt", padding=True).to(self.device)
            generated_ids = self.llm_model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.llm_tokenizer.eos_token_id
            )
            for i, out_ids in enumerate(generated_ids):
                input_len = inputs.input_ids[i].shape[0]
                text = self.llm_tokenizer.decode(out_ids[input_len:], skip_special_tokens=True).strip()
                descriptions.append(text)
        return descriptions

    @torch.no_grad()
    def generate_music_batch(
        self, 
        descriptions: List[str], 
        file_paths: List[str],
        max_new_tokens: int = 512, 
        output_dir: str = "./outputs",
    ) -> List[str]:
        os.makedirs(output_dir, exist_ok=True)
        inputs = self.musicgen_processor(
            text=descriptions,
            padding=True,
            return_tensors="pt"
        ).to(self.device)

        audio_outputs = self.musicgen_model.generate(**inputs, max_new_tokens=max_new_tokens)
        sampling_rate = self.musicgen_model.config.audio_encoder.sampling_rate
        saved_paths = []

        for idx, audio in enumerate(audio_outputs):
            # float32 before converting to np
            audio_data = audio[0].cpu().to(torch.float32).numpy()
            target_path = file_paths[idx]
            
            if not os.path.isabs(target_path):
                out_path = os.path.join(output_dir, target_path)
            else:
                out_path = target_path

            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            scipy.io.wavfile.write(out_path, rate=sampling_rate, data=audio_data)
            saved_paths.append(out_path)
            
        return saved_paths

    def process_videos(
        self, 
        dataloader: DataLoader, 
        frames_per_video: int = 15, 
        duration_seconds: int = 10,
        output_dir: str = "./generated_music"
    ) -> Dict[str, Union[List[str], List[Dict]]]:
        """
        Main pipeline function processing (video_path, save_path) tuples via DataLoader.
        """
        print(f"\nProcessing {len(dataloader.dataset)} video(s) in {len(dataloader)} batch(es)...")
        
        all_metadata = []
        all_audio_paths = []

        for batch_idx, (video_paths_batch, save_paths_batch) in enumerate(tqdm(dataloader, desc="Video Batches")):
            batch_llm_prompts = []
            batch_metadata = []
            batch_save_paths = []

            for video_path, save_path in zip(video_paths_batch, save_paths_batch):
                try:
                    frames = self.sample_frames_from_video(video_path, num_frames=frames_per_video)
                except Exception as e:
                    print(f"\nSkipping {video_path} due to error: {e}")
                    continue

                captions = self.generate_captions_batch(frames)
                emotions = self.analyze_emotions_batch(frames)
                prompt = self.construct_llm_prompt(captions, emotions)
                
                batch_llm_prompts.append(prompt)
                batch_save_paths.append(save_path)
                batch_metadata.append({
                    "video_path": video_path,
                    "target_save_path": save_path,
                    "captions": captions,
                    "emotions": emotions,
                    "llm_prompt": prompt
                })

            if not batch_llm_prompts:
                continue

            music_descriptions = self.query_llm_batch(batch_llm_prompts)
            for meta, desc in zip(batch_metadata, music_descriptions):
                meta["music_description"] = desc

            max_tokens = int(duration_seconds * 50)
            audio_paths = self.generate_music_batch(
                music_descriptions, 
                file_paths=batch_save_paths,
                max_new_tokens=max_tokens, 
                output_dir=output_dir
            )

            for meta, audio_path in zip(batch_metadata, audio_paths):
                meta["audio_output_path"] = audio_path
                
            all_metadata.extend(batch_metadata)
            all_audio_paths.extend(audio_paths)

        return {
            "metadata": all_metadata,
            "audio_paths": all_audio_paths
        }


if __name__ == "__main__":
    save_folder = "/home/es119256/dados/xps/herrmann1/inference"
    eval_path = "/home/es119256/dados/xps/ossl/ossl_base"
    dataset_path = "/home/es119256/dados/datasets/vmdb/nintendo-snes-spc"
    batch_size = 16
    num_workers = 8

    pred_to_orig_csv_path = os.path.join(eval_path, 'eval_gen/pred_to_orig.csv')
    
    sample_videos = []
    if os.path.exists(pred_to_orig_csv_path):
        pred_to_orig_df = pd.read_csv(pred_to_orig_csv_path)
        rows = list(pred_to_orig_df.itertuples(index=False, name=None))
        # rows = rows[3900:] # to split across multiple GPUs
        print(f"from {rows[0]} to {rows[-1]}")
        for row in tqdm(rows, total=len(pred_to_orig_df)):
            video_path,y_pred_path,y_path,y_seek,genres,caption,pt_path = row

            base_name = os.path.basename(y_pred_path)
            save_path = os.path.join(save_folder, f"{base_name}.wav")

            if os.path.exists(save_path):
                continue

            sample_videos.append((video_path, save_path))
    else:
        print(f"Warning: File {pred_to_orig_csv_path} not found. Ensure path is correct.")

    dataset = VideosDataset(sample_videos)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    pipeline = Herrmann1Pipeline(
        use_openai=False, 
        # openai_api_key= "",
        llm_model_id="Qwen/Qwen2.5-7B-Instruct" 
    )

    results = pipeline.process_videos(
        dataloader=dataloader,
        frames_per_video=10,
        duration_seconds=30,
        output_dir=save_folder
    )

    print("\n--- Processing Complete ---")
    if results and "metadata" in results:
        for res in results["metadata"]:
            print(f"Video: {res['video_path']}")
            print(f"Music Prompt: {res['music_description']}")
            print(f"Saved Waveform: {res['audio_output_path']}\n")