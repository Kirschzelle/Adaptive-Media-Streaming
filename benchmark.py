import subprocess
import time
import json
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Optional
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from codecarbon import EmissionsTracker # type: ignore
from skimage.metrics import peak_signal_noise_ratio as psnr # type: ignore
from skimage.metrics import structural_similarity as ssim # type: ignore
import cv2 # type: ignore
import numpy as np # type: ignore
import create_benchmark_profiles as profile
from datetime import datetime

USE_SVT = True
USE_VPX = False

SAMPLE_PERCENTAGE = 0.02
SAMPLE_MIN = 5

VMAF_TIMEOUT = 60000

@dataclass
class EncodingConfig:
    name: str
    encoder: bool

    bitrate: Optional[str] = None
    crf: Optional[int] = None
    keyframe_interval: Optional[int] = None
    threads: Optional[int] = None
    
    # Encoder-specific speed/quality
    # For libvpx: cpu_used (0-5)
    # For SVT: preset (0-8)
    speed_preset: Optional[int] = None
    
    extra_params: Dict[str, str] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.bitrate is None and self.crf is None:
            raise ValueError(f"Config '{self.name}': Either bitrate or crf must be specified")
        
        if self.bitrate is not None and self.crf is not None:
            raise ValueError(f"Config '{self.name}': Cannot specify both bitrate and crf")

@dataclass
class BenchmarkResult:
    config_name: str
    encoding_time: float
    energy_consumed: float
    file_size: int
    psnr_value: float
    ssim_value: float
    vmaf_score: float
    video_index: int = 0
    bitrate: str = ""
    cpu_used: int = 0

    def to_dict(self):
        return asdict(self)

class VideoEncodingBenchmark:
    
    def __init__(self, input_dir: str, output_dir: str = "./results/benchmark_results", quick = False):
        self.input_dir = Path(input_dir)
        self.input_dir.mkdir(exist_ok=True)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.quick = quick
        print("Getting Videos")
        self.videos = self._get_videos()
        print("Getting Video Info")
        self.infos = self._get_video_infos()
        self.results: List[BenchmarkResult] = []

    def encode_videos(self, config:EncodingConfig):
        benchmarks = []
        for i in range(0,len(self.videos)):
            benchmark = self._encode_video(i, config)
            if benchmark is None:
                continue
            benchmarks.append(benchmark)

        return benchmarks
    
    def run_benchmarks(self, configs: List[EncodingConfig]):
        print(f"\nStarting benchmark with {len(configs)} configurations...")
        print(f"Results will be saved to: {self.output_dir}")
        
        for config in configs:
            try:
                self.encode_videos(config)
            except Exception as e:
                print(f"Failed to encode {config.name}: {e}")
                continue
        
        print(f"Benchmark completed! {len(self.results)} results collected.")

    def save_results(self, filename: str = "results.json"):
        output_path = self.output_dir / filename
        with open(output_path, 'w') as f:
            json.dump([r.to_dict() for r in self.results], f, indent=2)
        print(f"\nResults saved to: {output_path}")
    
    def generate_report(self):
        if not self.results:
            print("No results to report!")
            return

        df = pd.DataFrame([r.to_dict() for r in self.results])
        video_indices = df['video_index'].unique()

        if len(video_indices) > 1:
            df_aggregated = df.groupby('config_name').agg({
                'encoding_time': 'sum',
                'energy_consumed': 'sum',
                'file_size': 'sum',
                'psnr_value': 'mean',
                'ssim_value': 'mean',
                'vmaf_score': 'mean',
                'bitrate': 'first',
                'cpu_used': 'first'
            }).reset_index()

            print("\nGenerating aggregated report...")
            rankings, df_norm = self._create_rankings(df_aggregated)
            self._plot_results(df_norm, rankings, subfolder="aggregated")

            print(f"\nGenerating per-video reports for {len(video_indices)} videos...")
            for video_idx in video_indices:
                df_video = df[df['video_index'] == video_idx].copy()
                df_video = df_video.drop(columns=['video_index'])

                try:
                    rankings_video, df_norm_video = self._create_rankings(df_video)
                    self._plot_results(df_norm_video, rankings_video, subfolder=f"video_{video_idx}")
                    print(f"  Generated report for video {video_idx}")
                except Exception as e:
                    print(f"  Warning: Could not generate report for video {video_idx}: {e}")
        else:
            # Single video - generate report in root directory
            print("\nGenerating report...")
            df_single = df.drop(columns=['video_index'])
            rankings, df_norm = self._create_rankings(df_single)
            self._plot_results(df_norm, rankings)
        
    def _encode_video(self, video_index, config: EncodingConfig) -> BenchmarkResult:
        if config.encoder == USE_SVT:
            return None # type: ignore
        
        print(f"{config.name}:{self.videos[video_index]}")
        
        output_file = self.output_dir / f"{config.name}_{video_index}.webm"
        
        cmd = self._build_ffmpeg_command(self.videos[video_index], config, output_file)
        
        tracker = EmissionsTracker(
            project_name=f"video_encoding_{config.name}_{video_index}",
            output_dir=str(self.output_dir),
            log_level="error",
            save_to_file=False
        )
        
        tracker.start()
        start_time = time.time()
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            encoding_time = time.time() - start_time
            emissions = tracker.stop()
                        
        except subprocess.CalledProcessError as e:
            tracker.stop()
            print(f"[ERROR] Encoding failed: {e}")
            print(f"[ERROR] Error output: {e.stderr.decode()}")
            raise

        print("Finished encoding")
        
        file_size = output_file.stat().st_size

        print("Calculating quality metrics")
        
        psnr_val, ssim_val = self._calculate_quality_metrics(video_index, SAMPLE_PERCENTAGE, output_file)
        
        vmaf_score = None
        
        result = BenchmarkResult(
            config_name=config.name,
            encoding_time=encoding_time,
            energy_consumed=emissions,
            file_size=file_size,
            psnr_value=psnr_val,
            ssim_value=ssim_val,
            vmaf_score=vmaf_score, # type: ignore
            video_index=video_index,
        )
        
        self.results.append(result)
        
        return result

    def _get_videos(self):
        video_extensions = {".mp4", ".avi", ".mkv", ".mov", ".flv", ".wmv", ".webm", ".m4v"}
        
        video_files = [
            f for f in self.input_dir.iterdir() 
            if f.is_file() and f.suffix.lower() in video_extensions
        ]
        
        if not video_files:
            raise FileNotFoundError(f"No video files found in {self.input_dir}")
        
        if self.quick:
            min_frames = float("inf")
            quickest_video = None
            
            print("Quick mode: Finding video with least frames...")
            for video_path in video_files:
                print(str(video_path))
                try:
                    info = self._get_video_info(video_path)
                    frames = info["frames"]
                    
                    if frames < min_frames:
                        min_frames = frames
                        quickest_video = video_path
                except Exception as e:
                    print(f"  Warning: Could not read {video_path.name}: {e}")
                    continue
            
            if quickest_video is None:
                raise RuntimeError("Could not find any valid video files")
            
            print(f"Selected: {quickest_video.name} with {min_frames} frames")
            return [quickest_video]
        
        print(f"Found {len(video_files)} video file(s)")
        return video_files
    
    def _get_video_infos(self):
        if self.videos == None:
            return
        
        infos = []
        
        for video in self.videos:
            print(f"Getting info for video: {str(video)}")
            info = self._get_video_info(video)
            infos.append(info)

        return infos

    def _get_video_info(self, video_path: Path) -> Dict[str, Any]:
        cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-count_frames",
            "-show_entries", "stream=width,height,duration,nb_read_frames",
            "-of", "json",
            str(video_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        stream = data["streams"][0]
        
        return {
            "width": int(stream["width"]),
            "height": int(stream["height"]),
            "duration": float(stream.get("duration", 0)),
            "frames": int(stream.get("nb_read_frames", 0))
        }
    
    def _build_ffmpeg_command(self, video, config: EncodingConfig, output_file: Path) -> List[str]:
        cmd = [
            "ffmpeg",
            "-y",  # Overwrite output
            "-i", str(video)
        ]

        if config.encoder == USE_SVT:
            cmd.extend(["-c:v", "libsvt_vp9"])
        elif config.encoder == USE_VPX:
            cmd.extend(["-c:v", "libvpx-vp9"])
        else:
            raise ValueError(f"Unknown encoder: {config.encoder}")
        
        if config.crf is not None:
            if config.encoder == USE_VPX:
                cmd.extend(["-crf", str(config.crf), "-b:v", "0"])
            elif config.encoder == USE_SVT:
                cmd.extend(["-qp", str(config.crf), "-rc", "0"])
        elif config.bitrate is not None:
            cmd.extend(["-b:v", config.bitrate])
            if config.encoder == USE_SVT:
                cmd.extend(["-rc", "1"]) # Sets variable bitrate.
        else:
            raise ValueError("Either bitrate or crf must be specified")
        
        if config.speed_preset is not None:
            if config.encoder == USE_VPX:
                cmd.extend(["-cpu-used", str(min(5,max(0,config.speed_preset)))])
            elif config.encoder == USE_SVT:
                cmd.extend(["-preset", str(min(8,max(0,config.speed_preset)))])

        if config.keyframe_interval is not None:
            cmd.extend(["-g", str(config.keyframe_interval)])

        if config.threads is not None:
            cmd.extend(["-threads", str(config.threads)])
        else:
            cmd.extend(["-threads", "0"]) # 0 = All aviable threads.

        if config.encoder == USE_VPX:
            if "-row-mt" not in config.extra_params:
                cmd.extend(["-row-mt", "1"])
            if "-quality" not in config.extra_params:
                cmd.extend(["-quality", "good"])
        
        if config.extra_params:
            for key, value in config.extra_params.items():
                cmd.extend([key, str(value)])

        cmd.extend(["-an"]) # Remove audio.
    
        cmd.append(str(output_file))

        return cmd
    
    def _calculate_quality_metrics(self, video_index, sample_precentage, encoded_video: Path) -> tuple:
        total_frames = self.infos[video_index]['frames'] # type: ignore

        num_samples = min(total_frames, max(SAMPLE_MIN, (int)(total_frames * sample_precentage)))
        sample_indices = [int(i * total_frames / (num_samples + 1)) for i in range(1, num_samples + 1)]
        
        psnr_values = []
        ssim_values = []
        
        for idx in sample_indices:
            print(f"Extracting frame {idx}")
            ref_frame = self._extract_frame(self.videos[video_index], idx)
            enc_frame = self._extract_frame(encoded_video, idx)
            
            if ref_frame is not None and enc_frame is not None:
                if ref_frame.shape != enc_frame.shape:
                    enc_frame = cv2.resize(enc_frame, (ref_frame.shape[1], ref_frame.shape[0]))
                
                psnr_val = psnr(ref_frame, enc_frame, data_range=255)
                ssim_val = ssim(ref_frame, enc_frame, multichannel=True, channel_axis=2, data_range=255)
                
                psnr_values.append(psnr_val)
                ssim_values.append(ssim_val)
        
        return np.mean(psnr_values), np.mean(ssim_values)
    
    def _extract_frame(self, video_path: Path, frame_number: int) -> np.ndarray:
        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            return frame
        return None
    
    def _calculate_vmaf(self, video_index, encoded_video: Path):
        try:
            vmaf_log = self.output_dir / f"{encoded_video.stem}_vmaf.json"
            
            cmd = [
                'ffmpeg',
                '-i', str(encoded_video),
                '-i', str(self.videos[video_index]),
                '-lavfi', f'[0:v]setpts=PTS-STARTPTS[dist];[1:v]setpts=PTS-STARTPTS[ref];[dist][ref]libvmaf=log_path={vmaf_log}:log_fmt=json',
                '-f', 'null',
                '-'
            ]
            
            subprocess.run(cmd, check=True, capture_output=True, timeout=VMAF_TIMEOUT)
            
            with open(vmaf_log, 'r') as f:
                vmaf_data = json.load(f)
                return vmaf_data['pooled_metrics']['vmaf']['mean']
        
        except (subprocess.CalledProcessError, FileNotFoundError, KeyError, subprocess.TimeoutExpired):
            print("  (VMAF calculation skipped - libvmaf may not be available)")
            return None
        
    def _create_rankings(self, df: pd.DataFrame) -> tuple:
        rankings = {}
        
        rankings['psnr'] = df.nlargest(len(df), 'psnr_value')[['config_name', 'psnr_value']]
        rankings['ssim'] = df.nlargest(len(df), 'ssim_value')[['config_name', 'ssim_value']]
        
        if df['vmaf_score'].notna().any():
            rankings['vmaf'] = df.nlargest(len(df), 'vmaf_score')[['config_name', 'vmaf_score']]
        
        df_norm = df.copy()
        
        df_norm['psnr_norm'] = (df['psnr_value'] - df['psnr_value'].min()) / (df['psnr_value'].max() - df['psnr_value'].min())
        df_norm['ssim_norm'] = (df['ssim_value'] - df['ssim_value'].min()) / (df['ssim_value'].max() - df['ssim_value'].min())
        
        if df['vmaf_score'].notna().any():
            df_norm['vmaf_norm'] = (df['vmaf_score'] - df['vmaf_score'].min()) / (df['vmaf_score'].max() - df['vmaf_score'].min())
            df_norm['quality_score'] = (0.4 * df_norm['vmaf_norm'] + 
                                    0.3 * df_norm['psnr_norm'] + 
                                    0.3 * df_norm['ssim_norm'])
        else:
            df_norm['quality_score'] = (0.5 * df_norm['psnr_norm'] + 
                                    0.5 * df_norm['ssim_norm'])
        
        rankings['quality'] = df_norm.nlargest(len(df), 'quality_score')[['config_name', 'quality_score', 'psnr_value', 'ssim_value', 'vmaf_score']]
        
        rankings['energy'] = df.nsmallest(len(df), 'energy_consumed')[['config_name', 'energy_consumed', 'encoding_time']]
        
        rankings['speed'] = df.nsmallest(len(df), 'encoding_time')[['config_name', 'encoding_time', 'file_size']]
        
        rankings['filesize'] = df.nsmallest(len(df), 'file_size')[['config_name', 'file_size', 'psnr_value']]
        
        df_norm['quality_per_mb'] = df_norm['quality_score'] / (df['file_size'] / (1024**2))  # Convert bytes to MiB
        rankings['compression'] = df_norm.nlargest(len(df), 'quality_per_mb')[['config_name', 'file_size', 'quality_score', 'quality_per_mb']]
        
        df_norm['time_norm'] = 1 - ((df['encoding_time'] - df['encoding_time'].min()) / (df['encoding_time'].max() - df['encoding_time'].min()))
        df_norm['energy_norm'] = 1 - ((df['energy_consumed'] - df['energy_consumed'].min()) / (df['energy_consumed'].max() - df['energy_consumed'].min()))
        df_norm['size_norm'] = 1 - ((df['file_size'] - df['file_size'].min()) / (df['file_size'].max() - df['file_size'].min()))
        
        df_norm['practical_score'] = (0.35 * df_norm['quality_score'] + 
                                    0.25 * df_norm['time_norm'] + 
                                    0.20 * df_norm['energy_norm'] +
                                    0.20 * df_norm['size_norm'])
        
        rankings['practical'] = df_norm.nlargest(len(df), 'practical_score')[['config_name', 'practical_score', 'psnr_value', 'encoding_time', 'energy_consumed', 'file_size']]
        
        return rankings, df_norm
    
    def _plot_results(self, df: pd.DataFrame, rankings: Dict[str, pd.DataFrame], subfolder: str = ""):
        sns.set_style("whitegrid")

        # Create subfolder if specified
        if subfolder:
            plot_dir = self.output_dir / subfolder
            plot_dir.mkdir(exist_ok=True)
        else:
            plot_dir = self.output_dir
        
        plt.figure(figsize=(20, 14))
        
        ax1 = plt.subplot(3, 4, 1)
        scatter1 = ax1.scatter(
            df['encoding_time'],
            df['psnr_value'], 
            s=df['energy_consumed']*10000, 
            c=df['file_size']/((2^10)^3), 
            cmap='viridis', 
            alpha=0.6
            )
        ax1.set_xlabel('Encoding Time (seconds)')
        ax1.set_ylabel('PSNR (dB)')
        ax1.set_title('Quality vs Speed\n(bubble size = energy, color = file size MB)')
        plt.colorbar(scatter1, ax=ax1, label='File Size (MB)')
        
        for _, row in df.iterrows():
            ax1.annotate(
                row['config_name'],  # type: ignore
                (row['encoding_time'], row['psnr_value']), # type: ignore
                fontsize=7, alpha=0.7)
        
        ax2 = plt.subplot(3, 4, 2)
        top_energy = rankings['energy'].head(10)
        bars = ax2.barh(range(len(top_energy)), top_energy['energy_consumed'])
        ax2.set_yticks(range(len(top_energy)))
        ax2.set_yticklabels(top_energy['config_name'], fontsize=8)
        ax2.set_xlabel('Energy Consumed (kWh)')
        ax2.set_title('Top 10 Energy Efficient')
        ax2.invert_yaxis()
        colors = plt.cm.RdYlGn_r(top_energy['energy_consumed'] / top_energy['energy_consumed'].max()) # type: ignore
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        ax3 = plt.subplot(3, 4, 3)
        scatter3 = ax3.scatter(df['file_size'] / ((2^10)^3), df['psnr_value'], 
                            c=df['encoding_time'], s=100, alpha=0.6, cmap='plasma')
        ax3.set_xlabel('File Size (MB)')
        ax3.set_ylabel('PSNR (dB)')
        ax3.set_title('Compression Efficiency\n(color = encoding time)')
        plt.colorbar(scatter3, ax=ax3, label='Time (s)')
        
        for _, row in df.iterrows():
            ax3.annotate(row['config_name'],  # type: ignore
                        (row['file_size'] / ((2^10)^3), row['psnr_value']), # type: ignore
                        fontsize=7, alpha=0.7)
        
        ax4 = plt.subplot(3, 4, 4)
        top_quality = rankings['quality'].head(10)
        ax4.barh(range(len(top_quality)), top_quality['quality_score'], color='skyblue')
        ax4.set_yticks(range(len(top_quality)))
        ax4.set_yticklabels(top_quality['config_name'], fontsize=8)
        ax4.set_xlabel('Quality Score')
        ax4.set_title('Top 10 by Combined Quality')
        ax4.invert_yaxis()
        
        ax5 = plt.subplot(3, 4, 5)
        top_psnr = rankings['psnr'].head(10)
        ax5.barh(range(len(top_psnr)), top_psnr['psnr_value'], color='lightcoral')
        ax5.set_yticks(range(len(top_psnr)))
        ax5.set_yticklabels(top_psnr['config_name'], fontsize=8)
        ax5.set_xlabel('PSNR (dB)')
        ax5.set_title('Top 10 by PSNR')
        ax5.invert_yaxis()
        
        ax6 = plt.subplot(3, 4, 6)
        top_ssim = rankings['ssim'].head(10)
        ax6.barh(range(len(top_ssim)), top_ssim['ssim_value'], color='lightgreen')
        ax6.set_yticks(range(len(top_ssim)))
        ax6.set_yticklabels(top_ssim['config_name'], fontsize=8)
        ax6.set_xlabel('SSIM')
        ax6.set_title('Top 10 by SSIM')
        ax6.invert_yaxis()
        
        ax7 = plt.subplot(3, 4, 7)
        if 'vmaf' in rankings:
            top_vmaf = rankings['vmaf'].head(10)
            ax7.barh(range(len(top_vmaf)), top_vmaf['vmaf_score'], color='plum')
            ax7.set_yticks(range(len(top_vmaf)))
            ax7.set_yticklabels(top_vmaf['config_name'], fontsize=8)
            ax7.set_xlabel('VMAF Score')
            ax7.set_title('Top 10 by VMAF')
            ax7.invert_yaxis()
        else:
            ax7.text(0.5, 0.5, 'VMAF not available', 
                    ha='center', va='center', transform=ax7.transAxes)
            ax7.set_title('VMAF Rankings')
        
        ax8 = plt.subplot(3, 4, 8)
        top_speed = rankings['speed'].head(10)
        ax8.barh(range(len(top_speed)), top_speed['encoding_time'], color='orange')
        ax8.set_yticks(range(len(top_speed)))
        ax8.set_yticklabels(top_speed['config_name'], fontsize=8)
        ax8.set_xlabel('Encoding Time (seconds)')
        ax8.set_title('Top 10 Fastest')
        ax8.invert_yaxis()
        
        ax9 = plt.subplot(3, 4, 9)
        top_filesize = rankings['filesize'].head(10)
        ax9.barh(range(len(top_filesize)), top_filesize['file_size'] / ((2^10)^3), color='gold')
        ax9.set_yticks(range(len(top_filesize)))
        ax9.set_yticklabels(top_filesize['config_name'], fontsize=8)
        ax9.set_xlabel('File Size (MB)')
        ax9.set_title('Top 10 Smallest Files')
        ax9.invert_yaxis()
        
        ax10 = plt.subplot(3, 4, 10)
        top_compression = rankings['compression'].head(10)
        ax10.barh(range(len(top_compression)), top_compression['quality_per_mb'], color='cyan')
        ax10.set_yticks(range(len(top_compression)))
        ax10.set_yticklabels(top_compression['config_name'], fontsize=8)
        ax10.set_xlabel('Quality per MB')
        ax10.set_title('Top 10 Compression Efficiency')
        ax10.invert_yaxis()
        
        ax11 = plt.subplot(3, 4, 11)
        top_practical = rankings['practical'].head(10)
        ax11.barh(range(len(top_practical)), top_practical['practical_score'], color='mediumpurple')
        ax11.set_yticks(range(len(top_practical)))
        ax11.set_yticklabels(top_practical['config_name'], fontsize=8)
        ax11.set_xlabel('Practical Score')
        ax11.set_title('Top 10 Overall (Balanced)')
        ax11.invert_yaxis()
        
        ax12 = plt.subplot(3, 4, 12)

        top_5 = rankings['practical'].head(5)
        metrics = ['quality_score', 'time_norm', 'energy_norm', 'size_norm']
        x = range(len(metrics))
        
        for _, row_idx in enumerate(top_5.index):
            values = [df.loc[row_idx, metric] for metric in ['quality_score', 'time_norm', 'energy_norm', 'size_norm']]
            ax12.plot(x, values, marker='o', label=df.loc[row_idx, 'config_name'], linewidth=2)
        
        ax12.set_xticks(x)
        ax12.set_xticklabels(['Quality', 'Speed', 'Energy', 'Size'], fontsize=8)
        ax12.set_ylabel('Normalized Score (0-1)')
        ax12.set_title('Top 5 Multi-Metric Comparison')
        ax12.legend(fontsize=7, loc='best')
        ax12.grid(True, alpha=0.3)
        ax12.set_ylim(0, 1.1)
        
        plt.tight_layout()

        output_path = plot_dir / "benchmark_results.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        self._create_detailed_plots(df, rankings, plot_dir)

    def _create_detailed_plots(self, df: pd.DataFrame, rankings: Dict[str, pd.DataFrame], plot_dir: Path):

        _, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        df_sorted = df.sort_values('psnr_value', ascending=False)
        axes[0, 0].bar(range(len(df_sorted)), df_sorted['psnr_value'], color='steelblue')
        axes[0, 0].set_xticks(range(len(df_sorted)))
        axes[0, 0].set_xticklabels(df_sorted['config_name'], rotation=45, ha='right', fontsize=8)
        axes[0, 0].set_ylabel('PSNR (dB)')
        axes[0, 0].set_title('PSNR by Configuration')
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        df_sorted = df.sort_values('ssim_value', ascending=False)
        axes[0, 1].bar(range(len(df_sorted)), df_sorted['ssim_value'], color='seagreen')
        axes[0, 1].set_xticks(range(len(df_sorted)))
        axes[0, 1].set_xticklabels(df_sorted['config_name'], rotation=45, ha='right', fontsize=8)
        axes[0, 1].set_ylabel('SSIM')
        axes[0, 1].set_title('SSIM by Configuration')
        axes[0, 1].grid(axis='y', alpha=0.3)
        
        if df['vmaf_score'].notna().any():
            df_sorted = df[df['vmaf_score'].notna()].sort_values('vmaf_score', ascending=False) # type: ignore
            axes[1, 0].bar(range(len(df_sorted)), df_sorted['vmaf_score'], color='coral')
            axes[1, 0].set_xticks(range(len(df_sorted)))
            axes[1, 0].set_xticklabels(df_sorted['config_name'], rotation=45, ha='right', fontsize=8)
            axes[1, 0].set_ylabel('VMAF Score')
            axes[1, 0].set_title('VMAF by Configuration')
            axes[1, 0].grid(axis='y', alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'VMAF not available', 
                        ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('VMAF by Configuration')
        
        df_sorted = df.sort_values('quality_score', ascending=False)
        axes[1, 1].bar(range(len(df_sorted)), df_sorted['quality_score'], color='mediumpurple')
        axes[1, 1].set_xticks(range(len(df_sorted)))
        axes[1, 1].set_xticklabels(df_sorted['config_name'], rotation=45, ha='right', fontsize=8)
        axes[1, 1].set_ylabel('Combined Quality Score')
        axes[1, 1].set_title('Overall Quality Score by Configuration')
        axes[1, 1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        output_path = plot_dir / "quality_metrics_detailed.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        _, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        df_sorted = df.sort_values('encoding_time')
        axes[0, 0].bar(range(len(df_sorted)), df_sorted['encoding_time'], color='orange')
        axes[0, 0].set_xticks(range(len(df_sorted)))
        axes[0, 0].set_xticklabels(df_sorted['config_name'], rotation=45, ha='right', fontsize=8)
        axes[0, 0].set_ylabel('Time (seconds)')
        axes[0, 0].set_title('Encoding Time by Configuration')
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        df_sorted = df.sort_values('energy_consumed')
        axes[0, 1].bar(range(len(df_sorted)), df_sorted['energy_consumed'] * 1000, color='crimson')
        axes[0, 1].set_xticks(range(len(df_sorted)))
        axes[0, 1].set_xticklabels(df_sorted['config_name'], rotation=45, ha='right', fontsize=8)
        axes[0, 1].set_ylabel('Energy (Wh)')
        axes[0, 1].set_title('Energy Consumption by Configuration')
        axes[0, 1].grid(axis='y', alpha=0.3)
        
        df_sorted = df.sort_values('file_size')
        axes[1, 0].bar(range(len(df_sorted)), df_sorted['file_size'] / ((2^10)^3), color='gold')
        axes[1, 0].set_xticks(range(len(df_sorted)))
        axes[1, 0].set_xticklabels(df_sorted['config_name'], rotation=45, ha='right', fontsize=8)
        axes[1, 0].set_ylabel('File Size (MB)')
        axes[1, 0].set_title('File Size by Configuration')
        axes[1, 0].grid(axis='y', alpha=0.3)
        
        df_sorted = df.sort_values('quality_per_mb', ascending=False)
        axes[1, 1].bar(range(len(df_sorted)), df_sorted['quality_per_mb'], color='cyan')
        axes[1, 1].set_xticks(range(len(df_sorted)))
        axes[1, 1].set_xticklabels(df_sorted['config_name'], rotation=45, ha='right', fontsize=8)
        axes[1, 1].set_ylabel('Quality per MB')
        axes[1, 1].set_title('Compression Efficiency by Configuration')
        axes[1, 1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        output_path = plot_dir / "performance_metrics_detailed.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        _, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        axes[0, 0].scatter(df['encoding_time'], df['quality_score'], s=150, alpha=0.6, c=df['energy_consumed'], cmap='plasma')
        axes[0, 0].set_xlabel('Encoding Time (seconds)')
        axes[0, 0].set_ylabel('Quality Score')
        axes[0, 0].set_title('Quality vs Speed Tradeoff')
        for _, row in df.iterrows():
            axes[0, 0].annotate(row['config_name'], (row['encoding_time'], row['quality_score']), fontsize=7, alpha=0.7)
        
        axes[0, 1].scatter(df['energy_consumed'], df['quality_score'], s=150, alpha=0.6, c=df['encoding_time'], cmap='viridis')
        axes[0, 1].set_xlabel('Energy Consumed (kWh)')
        axes[0, 1].set_ylabel('Quality Score')
        axes[0, 1].set_title('Quality vs Energy Tradeoff')
        for _, row in df.iterrows():
            axes[0, 1].annotate(row['config_name'], (row['energy_consumed'], row['quality_score']), fontsize=7, alpha=0.7)
        
        axes[1, 0].scatter(df['file_size'] / ((2^10)^3), df['quality_score'], s=150, alpha=0.6, c=df['encoding_time'], cmap='coolwarm')
        axes[1, 0].set_xlabel('File Size (MB)')
        axes[1, 0].set_ylabel('Quality Score')
        axes[1, 0].set_title('Quality vs File Size Tradeoff')
        for _, row in df.iterrows():
            axes[1, 0].annotate(row['config_name'], (row['file_size'] / ((2^10)^3), row['quality_score']), fontsize=7, alpha=0.7)
        
        axes[1, 1].scatter(df['encoding_time'], df['energy_consumed'], s=150, alpha=0.6, c=df['quality_score'], cmap='RdYlGn')
        axes[1, 1].set_xlabel('Encoding Time (seconds)')
        axes[1, 1].set_ylabel('Energy Consumed (kWh)')
        axes[1, 1].set_title('Energy vs Time Correlation')
        for _, row in df.iterrows():
            axes[1, 1].annotate(row['config_name'], (row['encoding_time'], row['energy_consumed']), fontsize=7, alpha=0.7)
        
        plt.tight_layout()
        output_path = plot_dir / "tradeoff_analysis.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()



def main():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f'./benchmark_results_{timestamp}'
    
    benchmark = VideoEncodingBenchmark(str(Path("./BenchmarkVideos")), output_dir, False)
    
    configs = profile.create()
    
    print(f"\nWill test {len(configs)} configurations:")
    for config in configs:
        print(f"  - {config.name}")
    
    benchmark.run_benchmarks(configs)
    
    benchmark.save_results()
    
    benchmark.generate_report()

if __name__ == "__main__":
    main()