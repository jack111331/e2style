import torch
from torch import nn
from models.encoders import backbone_encoders
from configs.paths_config import model_paths
import pickle
import numpy as np
from models.stylegan2.modified_sg2 import Generator

def get_keys(d, name):
	if 'state_dict' in d:
		d = d['state_dict']
	d_filt = {k[len(name) + 1:]: v for k, v in d.items() if (k[:len(name)] == name) and (k[len(name)] != '_')}
	return d_filt


class E2Style(nn.Module):

	def __init__(self, opts):
		super(E2Style, self).__init__()
		self.set_opts(opts)
		self.stage = self.opts.training_stage if self.opts.is_training is True else self.opts.stage
		self.encoder_firststage = backbone_encoders.BackboneEncoderFirstStage(50, 'ir_se', self.opts)

		if self.stage > 1:
			self.encoder_refinestage_list = nn.ModuleList([backbone_encoders.BackboneEncoderRefineStage(50, 'ir_se', self.opts) for i in range(self.stage-1)])

		with open(self.opts.stylegan_weights, 'rb') as f:
			networks = pickle.Unpickler(f).load()

		# NOTE maybe we can get it from networks['G_ema']?
		mapping_kwargs = {}
		mapping_kwargs["num_layers"] = 8
		synthesis_kwargs = {}
		synthesis_kwargs["channel_base"] = 32768
		synthesis_kwargs["channel_max"] = 512
		synthesis_kwargs["num_fp16_res"] = 4
		synthesis_kwargs["conv_clamp"] = 256

		self.decoder = Generator(z_dim=512, c_dim=0, w_dim=512, img_resolution=1024, img_channels=3, square=False, mapping_kwargs=mapping_kwargs, synthesis_kwargs=synthesis_kwargs)
		self.decoder.load_state_dict(networks["G_ema"].state_dict(), strict=False)
		for param in self.decoder.parameters():
				param.requires_grad = False
		self.residue = backbone_encoders.ResidualEncoder() #Ec
		self.face_pool = torch.nn.AdaptiveAvgPool2d((256, 128))
		self.load_weights()


	def load_weights(self):
		if (self.opts.checkpoint_path is not None) and (not self.opts.is_training):
			if self.stage > self.opts.training_stage:
				raise ValueError(f'The stage must be no greater than {self.opts.training_stage} when testing!')
			print(f'Inference: Results are from Stage{self.stage}.', flush=True)
			print('Loading E2Style from checkpoint: {}'.format(self.opts.checkpoint_path), flush=True)
			ckpt = torch.load(self.opts.checkpoint_path, map_location='cpu')
			self.encoder_firststage.load_state_dict(get_keys(ckpt, 'encoder_firststage'), strict=True)
			self.residue.load_state_dict(get_keys(ckpt, 'residue'), strict=True)
			if self.stage > 1:
				for i in range(self.stage-1):
					self.encoder_refinestage_list[i].load_state_dict(get_keys(ckpt, f'encoder_refinestage_list.{i}'), strict=True)
			self.__load_latent_avg(ckpt)
		elif (self.opts.checkpoint_path is not None) and self.opts.is_training:
			print(f'Train: The {self.stage}-th encoder of E2Style is to be trained.', flush=True)
			print('Loading previous encoders and decoder from checkpoint: {}'.format(self.opts.checkpoint_path), flush=True)
			ckpt = torch.load(self.opts.checkpoint_path, map_location='cpu')
			self.encoder_firststage.load_state_dict(get_keys(ckpt, 'encoder_firststage'), strict=True)
			self.residue.load_state_dict(get_keys(ckpt, 'residue'), strict=True)
			if self.stage > 2:
				for i in range(self.stage-2):
					self.encoder_refinestage_list[i].load_state_dict(get_keys(ckpt, f'encoder_refinestage_list.{i}'), strict=True)
			print(f'Loading the {self.stage}-th encoder weights from irse50!')
			# encoder_ckpt = torch.load(model_paths['ir_se50'])
			# encoder_ckpt = {k: v for k, v in encoder_ckpt.items() if "input_layer" not in k}
			# self.encoder_refinestage_list[self.stage-2].load_state_dict(encoder_ckpt, strict=False)
			self.__load_latent_avg(ckpt)
		elif (self.opts.checkpoint_path is None) and (self.stage==1) and self.opts.is_training:
			print(f'Train: The 1-th encoder of E2Style is to be trained.', flush=True)
			# print('Loading encoders weights from irse50!')
			# encoder_ckpt = torch.load(model_paths['ir_se50'])
			# if input to encoder is not an RGB image, do not load the input layer weights
			# if self.opts.label_nc != 0:
			# 	encoder_ckpt = {k: v for k, v in encoder_ckpt.items() if "input_layer" not in k}
			# self.encoder_firststage.load_state_dict(encoder_ckpt, strict=False)
			print('Loading decoder weights from pretrained!')
			z_samples = np.random.RandomState(123).randn(10000, self.decoder.z_dim)
			w_samples = self.decoder.mapping(torch.from_numpy(z_samples), None)  # [N, L, C]
			w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32)  # [N, 1, C]
			w_avg = np.mean(w_samples, axis=0, keepdims=True)  # [1, 1, C]

			self.latent_avg = torch.from_numpy(w_avg).to('cuda:0')
			# z_samples = np.random.RandomState(123).randn(200, self.decoder.z_dim)
			# w_samples = self.decoder.mapping(torch.from_numpy(z_samples), None)  # [N, L, C]
			# sample_male = [2, 4, 5, 11, 22, 24, 33, 34, 37, 41, 46, 54, 62, 66, 68, 73, 75, 76, 77, 83, 84, 85, 88, 111, 113, 117, 119, 124
			# , 129, 136, 155, 162, 176, 180, 184, 187, 188, 192, 199]
			# w_samples = w_samples[sample_male, :1, :].cpu().numpy().astype(np.float32)  # [N, 1, C]
			# w_avg = np.mean(w_samples, axis=0, keepdims=True)  # [1, 1, C]
			# self.latent_avg = torch.from_numpy(w_avg).to('cuda:0')


			# if self.opts.learn_in_w:
			# 	self.__load_latent_avg(ckpt, repeat=1)
			# else:
			# 	self.__load_latent_avg(ckpt, repeat=18)		


	def forward(self, x, resize=True, input_code=False, randomize_noise=True, return_latents=False):

		stage_output_list = []
		if input_code:
			codes = x
		else:
			codes = self.encoder_firststage(x)
			if self.opts.start_from_latent_avg:
				if self.opts.learn_in_w:
					codes = codes + self.latent_avg.repeat(codes.shape[0], 1)
				else: 
					codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)
		input_is_latent = not input_code
		first_stage_output = self.decoder.synthesis(codes, noise_mode='random')
		stage_output_list.append(first_stage_output)

		imgs_ = nn.functional.interpolate(torch.clamp(stage_output_list[-1], -1., 1.), size=(256, 128) , mode='bilinear') 
		delta = x - imgs_
		conditions = self.residue(delta)
		high_rate_output = self.decoder.synthesis(codes, conditions=conditions, noise_mode='random')

		if self.stage > 1:
			for i in range(self.stage-1):
				codes = codes + self.encoder_refinestage_list[i](x, self.face_pool(stage_output_list[i]))
				refine_stage_output = self.decoder.synthesis(codes, noise_mode='random')
				stage_output_list.append(refine_stage_output)

		if resize: 
			images = self.face_pool(high_rate_output)
		else:
			images = high_rate_output

		if return_latents:
			return images, codes
		else:
			return images

	def set_opts(self, opts):
		self.opts = opts

	def __load_latent_avg(self, ckpt, repeat=None): 
		if 'latent_avg' in ckpt:
			self.latent_avg = ckpt['latent_avg'].to(self.opts.device)
			if repeat is not None:
				self.latent_avg = self.latent_avg.repeat(repeat, 1)
		else:
			self.latent_avg = None
