"""Convert the verified public BGE .bin to safetensors for stable torch 2.5 compatibility."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiment_io import file_hash,read_json,write_json


def main():
    import torch
    from safetensors.torch import save_file
    root=Path(__file__).resolve().parents[1];folder=root/'assets/models/bge-m3'
    source=folder/'pytorch_model.bin';target=folder/'model.safetensors';record=folder/'safetensors_conversion.json'
    if record.exists()and target.exists()and read_json(record)['target_sha256']==file_hash(target):
        print('BGE_SAFETENSORS_ALREADY_READY');return
    manifest=read_json(folder/'source_manifest.json');source_hash=file_hash(source)
    if source_hash!=manifest['sha256']['pytorch_model.bin']:raise ValueError('Public source checksum mismatch')
    state=torch.load(source,map_location='cpu',weights_only=True)
    tensors={k:v.contiguous().clone()for k,v in state.items()}
    temporary=folder/'model.safetensors.partial'
    save_file(tensors,str(temporary),metadata={'format':'pt'})
    temporary.replace(target)
    write_json(record,{'source_sha256':source_hash,'target_sha256':file_hash(target),'torch':torch.__version__,'conversion':'tensor values unchanged; contiguous clone; metadata format=pt'})
    print('BGE_SAFETENSORS_READY',flush=True)


if __name__=='__main__':main()
