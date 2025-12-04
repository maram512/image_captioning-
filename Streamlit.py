# app.py
import streamlit as st
import torch
import math
from PIL import Image
from torchvision import transforms
from transformers import ViTModel

# --------------------------
# 1) Device
# --------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------
# 2) Positional Encoding
# --------------------------
class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe = pe.unsqueeze(1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(0)
        x = x + self.pe[:seq_len]
        return x

# --------------------------
# 3) Image Captioning Model
# --------------------------
class ImageCaptioningModel(torch.nn.Module):
    def __init__(self, vocab_size, decoder_dim=512, nhead=8, num_layers=4, vit_model_name="google/vit-base-patch16-224-in21k", dropout=0.1):
        super().__init__()
        self.vit = ViTModel.from_pretrained(vit_model_name)
        vit_hidden = self.vit.config.hidden_size
        self.enc_proj = torch.nn.Linear(vit_hidden, decoder_dim)
        self.token_emb = torch.nn.Embedding(vocab_size, decoder_dim)
        self.pos_enc = PositionalEncoding(decoder_dim, max_len=100)
        decoder_layer = torch.nn.TransformerDecoderLayer(d_model=decoder_dim, nhead=nhead, dim_feedforward=decoder_dim*4, dropout=dropout)
        self.transformer_decoder = torch.nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.generator = torch.nn.Linear(decoder_dim, vocab_size)
        self.dropout = torch.nn.Dropout(dropout)
        self.decoder_dim = decoder_dim

    def encode_image(self, images):
        vit_outputs = self.vit(pixel_values=images)
        enc = vit_outputs.last_hidden_state
        enc = self.enc_proj(enc)
        return enc

    def decode_greedy(self, images, max_len=30, start_idx=1, end_idx=2):
        self.eval()
        with torch.no_grad():
            memory = self.encode_image(images).permute(1,0,2)
            batch = images.size(0)
            generated = torch.full((batch,1), start_idx, dtype=torch.long, device=images.device)
            for t in range(max_len-1):
                tgt_emb = self.token_emb(generated).permute(1,0,2)
                tgt_emb = self.pos_enc(tgt_emb)
                tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(tgt_emb.size(0)).to(images.device)
                out = self.transformer_decoder(tgt_emb, memory, tgt_mask=tgt_mask)
                out_last = out[-1]
                logits = self.generator(out_last)
                next_word = logits.argmax(dim=-1).unsqueeze(1)
                generated = torch.cat([generated, next_word], dim=1)
            # trim at end_idx
            results = []
            for i in range(batch):
                seq = generated[i].tolist()
                if end_idx in seq:
                    seq = seq[:seq.index(end_idx)+1]
                results.append(seq)
            return results

# --------------------------
# 4) Load checkpoint
# --------------------------
from huggingface_hub import hf_hub_download

checkpoint_path = hf_hub_download(
    repo_id="maram5/image-captioning-model",
    filename="caption_model_deploy (1).pt"
)
checkpoint = torch.load(checkpoint_path, map_location=device)
vocab = checkpoint["vocab"]
vocab_size = len(vocab)

model = ImageCaptioningModel(vocab_size=vocab_size).to(device)
model.load_state_dict(checkpoint["model_state"], strict=False)
model.eval()

# --------------------------
# 5) Transform
# --------------------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

# --------------------------
# 6) Streamlit UI
# --------------------------
st.title("📸 Image Captioning")

uploaded = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded Image", width=400)

    tensor = transform(img).unsqueeze(0).to(device)

    # --- Determine correct start/end tokens safely ---
    start_idx = vocab.get("<start>", vocab.get("<SOS>", 1))
    end_idx   = vocab.get("<end>", vocab.get("<EOS>", 2))

    with st.spinner("Generating caption..."):
        output_ids = model.decode_greedy(
            tensor,
            max_len=30,
            start_idx=start_idx,
            end_idx=end_idx
        )

        inv_vocab = {v: k for k, v in vocab.items()}
        words = [inv_vocab[idx] for idx in output_ids[0] if idx in inv_vocab]

        # --- REMOVE special tokens ---
        words = [
            w for w in words
            if w not in ["<start>", "<end>", "<SOS>", "<EOS>"]
        ]

        caption = " ".join(words)

    st.subheader("Caption:")
    st.write(caption)

