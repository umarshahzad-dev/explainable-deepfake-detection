import unittest
from unittest.mock import patch
import numpy as np
import torch
from tests.support import module


class MixUpTests(unittest.TestCase):
    def test_convex_mix_and_soft_label_sum(self):
        api=module(self,'data')
        images=torch.stack([torch.zeros(3,8,8),torch.ones(3,8,8)])
        labels=torch.tensor([[0.],[1.]])
        with patch('numpy.random.beta',return_value=.25),patch('torch.randperm',return_value=torch.tensor([1,0])):
            mixed,targets=api.mixup_batch(images,labels,.2)
        self.assertTrue(torch.allclose(mixed[0],torch.full((3,8,8),.75)))
        self.assertTrue(torch.allclose(targets[:,0],torch.tensor([.75,.25])))
        self.assertEqual(float(targets.sum()),float(labels.sum()))

    def test_zero_alpha_returns_original_batch(self):
        api=module(self,'data');images=torch.rand(2,3,8,8);labels=torch.tensor([0.,1.])
        mixed,targets=api.mixup_batch(images,labels,0)
        self.assertIs(mixed,images);self.assertIs(targets,labels)

    def test_train_epoch_uses_soft_targets_only_when_enabled(self):
        api=module(self,'training')
        model=torch.nn.Sequential(torch.nn.Flatten(),torch.nn.Linear(3*8*8,1))
        optimizer=torch.optim.SGD(model.parameters(),lr=.01)
        loader=[(torch.rand(2,3,8,8),torch.tensor([0.,1.]))]
        with patch('deepfake_detection.training.trainer.mixup_batch',return_value=(loader[0][0],torch.tensor([[.3],[.7]]))) as mix:
            result=api.train_epoch(model,loader,optimizer,device='cpu',amp=False,recipe={'mixup_alpha':.2})
        mix.assert_called_once()
        self.assertEqual(result['labels'],[0,1])
        self.assertEqual(result['optimizer_updates'],1)


class CutMixTests(unittest.TestCase):
    def test_patch_area_controls_soft_target_and_preserves_inputs(self):
        api=module(self,'data')
        images=torch.stack([torch.zeros(3,8,8),torch.ones(3,8,8)])
        labels=torch.tensor([[0.],[1.]])
        with patch('numpy.random.beta',return_value=.25),patch('torch.randperm',return_value=torch.tensor([1,0])),patch('torch.randint',return_value=torch.tensor([4])):
            mixed,targets=api.cutmix_batch(images,labels,.2)
        self.assertAlmostEqual(float(mixed[0].mean()),float(targets[0]),places=6)
        self.assertEqual(float(targets.sum()),1)
        self.assertEqual(float(images[0].sum()),0)
        self.assertEqual(mixed.shape,images.shape)

    def test_cutmix_zero_alpha_is_exactly_disabled(self):
        images=torch.rand(2,3,8,8);labels=torch.tensor([0.,1.])
        output,targets=module(self,'data').cutmix_batch(images,labels,0)
        self.assertIs(output,images);self.assertIs(targets,labels)

    def test_training_dispatches_cutmix_and_rejects_conflicting_batch_mixers(self):
        api=module(self,'training')
        model=torch.nn.Sequential(torch.nn.Flatten(),torch.nn.Linear(3*8*8,1));optimizer=torch.optim.SGD(model.parameters(),lr=.01)
        batch=(torch.rand(2,3,8,8),torch.tensor([0.,1.]))
        with patch('deepfake_detection.training.trainer.cutmix_batch',return_value=(batch[0],torch.tensor([[.3],[.7]]))) as mix:
            api.train_epoch(model,[batch],optimizer,amp=False,recipe={'cutmix_alpha':.2})
        mix.assert_called_once()
        with self.assertRaisesRegex(ValueError,'MixUp.*CutMix'):
            api.train_epoch(model,[batch],optimizer,amp=False,recipe={'cutmix_alpha':.2,'mixup_alpha':.2})


class FourierMixTests(unittest.TestCase):
    def test_amplitude_blend_preserves_source_phase_and_gradients(self):
        api=module(self,'data')
        a=torch.rand(3,8,8,requires_grad=True);b=torch.rand(3,8,8)
        output=api.fouriermix(a,b,alpha=.2)
        source=torch.fft.fft2(a.detach());donor=torch.fft.fft2(b)
        expected=torch.fft.ifft2(((1-.2)*source.abs()+.2*donor.abs())*torch.exp(1j*torch.angle(source))).real
        self.assertTrue(torch.allclose(output,expected,atol=1e-6))
        output.square().mean().backward();self.assertTrue(torch.isfinite(a.grad).all())

    def test_zero_alpha_and_zero_probability_disable_fouriermix(self):
        api=module(self,'data');image=torch.rand(3,8,8);batch=image[None]
        self.assertIs(api.fouriermix(image,image,0),image)
        self.assertIs(api.fouriermix_batch(batch,0),batch)

    def test_training_fouriermix_dispatches_when_probability_positive(self):
        api=module(self,'training')
        model=torch.nn.Sequential(torch.nn.Flatten(),torch.nn.Linear(192,1));optimizer=torch.optim.SGD(model.parameters(),lr=.01)
        image=torch.rand(2,3,8,8)
        with patch('deepfake_detection.training.trainer.fouriermix_batch',return_value=image) as mix:
            api.train_epoch(model,[(image,torch.tensor([0.,1.]))],optimizer,amp=False,recipe={'fouriermix_prob':.3})
        mix.assert_called_once()


class FFTAmplitudeMixTests(unittest.TestCase):
    def test_phase_batch_mix_uses_soft_targets_and_is_config_dispatched(self):
        api=module(self,'training');image=torch.rand(2,3,8,8)
        model=torch.nn.Sequential(torch.nn.Flatten(),torch.nn.Linear(192,1));optimizer=torch.optim.SGD(model.parameters(),lr=.01)
        with patch('deepfake_detection.training.trainer.fft_amplitude_mix_batch',return_value=(image,torch.tensor([[.2],[.8]]))) as mixed:
            api.train_epoch(model,[(image,torch.tensor([0.,1.]))],optimizer,amp=False,recipe={'fft_amplitude_mix_prob':.3})
        mixed.assert_called_once()
    def test_endpoints_and_constant_amplitude_blend(self):
        api=module(self,'data');a=torch.ones(3,8,8);b=a*2
        self.assertIs(api.fft_amplitude_mix(a,b,0),a)
        self.assertTrue(torch.allclose(api.fft_amplitude_mix(a,b,1),b,atol=1e-6))
        self.assertTrue(torch.allclose(api.fft_amplitude_mix(a,b,.5),a*1.5,atol=1e-6))

    def test_phase_wrap_interpolates_near_pi_not_zero(self):
        api=module(self,'data')
        coordinate=torch.arange(8)*2*np.pi/8
        a=torch.cos(coordinate+np.pi-.1)[None,None,:].expand(1,8,8)
        b=torch.cos(coordinate-np.pi+.1)[None,None,:].expand(1,8,8)
        self.assertLess(float(api.fft_amplitude_mix(a,b,.5)[0,0,0]),-.9)

    def test_interpolated_phase_mix_has_finite_gradient(self):
        a=torch.rand(3,8,8,requires_grad=True);b=torch.rand_like(a)
        result=module(self,'data').fft_amplitude_mix(a,b,.5)
        result.square().mean().backward()
        self.assertTrue(torch.isfinite(a.grad).all())
        self.assertGreater(float(a.grad.abs().sum()),0)


class DCTMaskTests(unittest.TestCase):
    def test_zero_masks_are_disabled_and_odd_sizes_work(self):
        api=module(self,'data');image=torch.rand(3,7,7)
        self.assertIs(api.random_dct_mask(image,num_masks=0),image)
        result=api.random_dct_mask(image,num_masks=1,mask_size=2)
        self.assertEqual(result.shape,image.shape)
        self.assertTrue(torch.isfinite(result).all())

    def test_masking_is_seeded_and_changes_coefficients(self):
        api=module(self,'data');image=torch.rand(3,8,8)
        torch.manual_seed(42);first=api.random_dct_mask(image,num_masks=2,mask_size=2)
        torch.manual_seed(42);second=api.random_dct_mask(image,num_masks=2,mask_size=2)
        self.assertTrue(torch.equal(first,second));self.assertFalse(torch.allclose(first,image))

    def test_dct_roundtrip_and_unmasked_gradients(self):
        api=module(self,'data');image=torch.rand(3,7,7,requires_grad=True)
        coefficients=api.dct2(image)
        self.assertTrue(torch.allclose(api.idct2(coefficients),image,atol=1e-5))
        result=api.random_dct_mask(image,num_masks=1,mask_size=1)
        result.square().mean().backward();self.assertTrue(torch.isfinite(image.grad).all())


class JPEGRangeTests(unittest.TestCase):
    def test_robust_v2_uses_configured_quality_range(self):
        from PIL import Image
        api=module(self,'data')
        augment=api.Augment('robust_v2',{'jpeg_quality_range':[30,100]})
        with patch('random.random',side_effect=[1,1,1,1,0,1]),patch('random.randint',return_value=30) as quality:
            output=augment(Image.new('RGB',(32,32),'red'))
        quality.assert_called_once_with(30,100)
        self.assertEqual(output.size,(32,32));self.assertEqual(output.mode,'RGB')

    def test_jpeg_probability_zero_disables_compression(self):
        from PIL import Image
        api=module(self,'data')
        with patch('random.random',return_value=1),patch('deepfake_detection.data.datasets.jpeg_compress') as jpeg:
            api.Augment('robust_v2',{'jpeg_prob':0})(Image.new('RGB',(32,32)))
        jpeg.assert_not_called()


class ResolutionRangeTests(unittest.TestCase):
    def test_range_used_and_original_dimensions_restored(self):
        from PIL import Image
        api=module(self,'data');pattern=(np.indices((32,32)).sum(axis=0)%2*255).astype(np.uint8)
        image=Image.fromarray(np.repeat(pattern[:,:,None],3,axis=2))
        augment=api.Augment('robust_v2',{'rescale_range':[.75,.75]})
        with patch('random.random',side_effect=[1,1,1,1,1,0]),patch('random.uniform',return_value=.75) as scale:
            output=augment(image)
        scale.assert_called_once_with(.75,.75)
        self.assertEqual(output.size,image.size)
        self.assertFalse(np.array_equal(np.asarray(output),np.asarray(image)))

    def test_resolution_augmentation_can_be_disabled(self):
        from PIL import Image
        api=module(self,'data');image=Image.new('RGB',(32,32))
        self.assertIs(api.random_rescale(image,[.5,1],prob=0),image)
        self.assertIs(api.random_rescale(image,[1,1],prob=1),image)


class GaussianNoiseTests(unittest.TestCase):
    def test_seeded_normalized_noise_has_requested_std(self):
        api=module(self,'data');image=torch.zeros(3,64,64)
        torch.manual_seed(42);first=api.add_gaussian_noise(image,.02)
        torch.manual_seed(42);second=api.add_gaussian_noise(image,.02)
        self.assertTrue(torch.equal(first,second))
        self.assertAlmostEqual(float(first.std()),.02,delta=.002)
        self.assertAlmostEqual(float(first.mean()),0,delta=.002)

    def test_zero_noise_returns_original_and_values_are_not_pixel_clipped(self):
        api=module(self,'data');image=torch.full((3,8,8),2.)
        self.assertIs(api.add_gaussian_noise(image,0),image)
        self.assertGreater(float(api.add_gaussian_noise(image,.02).min()),1)

    def test_training_noise_is_config_controlled(self):
        api=module(self,'training');image=torch.rand(2,3,8,8)
        model=torch.nn.Sequential(torch.nn.Flatten(),torch.nn.Linear(192,1));optimizer=torch.optim.SGD(model.parameters(),lr=.01)
        with patch('deepfake_detection.training.trainer.add_gaussian_noise',return_value=image) as noise:
            api.train_epoch(model,[(image,torch.tensor([0.,1.]))],optimizer,amp=False,recipe={'gaussian_noise_std':.02})
        noise.assert_called_once_with(image,.02)


class SaltPepperTests(unittest.TestCase):
    def test_pixels_are_black_or_white_with_seeded_selection(self):
        from PIL import Image
        api=module(self,'data');image=Image.new('RGB',(32,32),(127,127,127))
        torch.manual_seed(42);first=api.add_salt_pepper(image,.5)
        torch.manual_seed(42);second=api.add_salt_pepper(image,.5)
        values=np.asarray(first)
        self.assertTrue(np.array_equal(values,np.asarray(second)))
        self.assertEqual(set(np.unique(values)),{0,127,255})
        self.assertTrue(np.array_equal(values[:,:,0],values[:,:,1]))

    def test_salt_pepper_zero_probability_returns_original(self):
        image=torch.full((3,8,8),.5)
        self.assertIs(module(self,'data').add_salt_pepper(image,0),image)


class SharpnessTests(unittest.TestCase):
    def test_sharpness_factor_range_changes_pixels_and_keeps_size(self):
        from PIL import Image
        api=module(self,'data')
        array=np.zeros((16,16,3),dtype=np.uint8);array[6:10,6:10]=128
        image=Image.fromarray(array)
        with patch('random.random',return_value=0),patch('random.uniform',return_value=2) as factor:
            output=api.random_sharpen(image,(.5,2),prob=1)
        factor.assert_called_once_with(.5,2)
        self.assertEqual(output.size,image.size)
        self.assertFalse(np.array_equal(np.asarray(output),array))

    def test_sharpness_zero_probability_returns_original(self):
        from PIL import Image
        image=Image.new('RGB',(16,16))
        self.assertIs(module(self,'data').random_sharpen(image,prob=0),image)


class ColorJitterTests(unittest.TestCase):
    def test_hue_and_saturation_ranges_are_used(self):
        from PIL import Image
        api=module(self,'data');image=Image.new('RGB',(16,16),(128,60,20))
        with patch('random.random',return_value=0),patch('random.uniform',side_effect=[.05,1.15]) as draw:
            output=api.random_color_jitter(image,hue=.05,sat=.15,prob=1)
        self.assertEqual(draw.call_args_list,[unittest.mock.call(-.05,.05),unittest.mock.call(.85,1.15)])
        self.assertNotEqual(output.getpixel((8,8)),image.getpixel((8,8)))
        self.assertEqual(output.mode,'RGB')

    def test_zero_probability_or_zero_jitter_is_disabled(self):
        from PIL import Image
        api=module(self,'data');image=Image.new('RGB',(16,16))
        self.assertIs(api.random_color_jitter(image,prob=0),image)
        self.assertIs(api.random_color_jitter(image,hue=0,sat=0,prob=1),image)


class GammaTests(unittest.TestCase):
    def test_gamma_range_changes_pixel_intensity(self):
        from PIL import Image
        api=module(self,'data');image=Image.new('RGB',(16,16),(128,128,128))
        with patch('random.random',return_value=0),patch('random.uniform',return_value=2) as gamma:
            output=api.random_gamma(image,(.8,2),prob=1)
        gamma.assert_called_once_with(.8,2)
        self.assertEqual(output.getpixel((8,8)),(64,64,64))

    def test_gamma_disable_paths_are_exact_noops(self):
        image=torch.full((3,8,8),.5);api=module(self,'data')
        self.assertIs(api.random_gamma(image,prob=0),image)
        self.assertIs(api.random_gamma(image,(1,1),prob=1),image)


class CoarseDropoutTests(unittest.TestCase):
    def test_hole_area_is_bounded_and_input_not_modified(self):
        from PIL import Image
        image=Image.new('RGB',(32,32),'white')
        with patch('random.random',return_value=0),patch('random.randint',return_value=0):
            output=module(self,'data').coarse_dropout(image,prob=1,num_holes=1,hole_ratio=.25)
        values=np.asarray(output)
        self.assertEqual(int((values[:,:,0]==0).sum()),64)
        self.assertEqual(image.getpixel((0,0)),(255,255,255))

    def test_zero_probability_or_holes_is_disabled(self):
        image=torch.ones(3,8,8);api=module(self,'data')
        self.assertIs(api.coarse_dropout(image,prob=0),image)
        self.assertIs(api.coarse_dropout(image,prob=1,num_holes=0),image)


class GridShuffleTests(unittest.TestCase):
    def test_tiles_are_permuted_without_losing_pixels_on_even_grid(self):
        import random
        api=module(self,'data');image=torch.arange(3*8*8,dtype=torch.float32).reshape(3,8,8)
        random.seed(42);first=api.random_grid_shuffle(image,grid_size=4,prob=1)
        random.seed(42);second=api.random_grid_shuffle(image,grid_size=4,prob=1)
        self.assertTrue(torch.equal(first,second));self.assertFalse(torch.equal(first,image))
        self.assertTrue(torch.equal(first.flatten().sort().values,image.flatten().sort().values))

    def test_odd_sizes_and_disable_switch(self):
        api=module(self,'data');image=torch.rand(3,7,7)
        self.assertEqual(api.random_grid_shuffle(image,grid_size=4,prob=1).shape,image.shape)
        self.assertIs(api.random_grid_shuffle(image,prob=0),image)


class PerspectiveTests(unittest.TestCase):
    def test_perspective_warps_corners_and_keeps_dimensions(self):
        from PIL import Image
        image=Image.new('RGB',(32,32),'white')
        with patch('random.random',return_value=0),patch('random.randint',return_value=6):
            output=module(self,'data').random_perspective(image,scale=.2,prob=1)
        self.assertEqual(output.size,image.size)
        self.assertEqual(output.getpixel((0,0)),(0,0,0))
        self.assertGreater(output.getpixel((16,16))[0],200)

    def test_perspective_disabled_or_zero_scale_is_noop(self):
        from PIL import Image
        image=Image.new('RGB',(32,32));api=module(self,'data')
        self.assertIs(api.random_perspective(image,prob=0),image)
        self.assertIs(api.random_perspective(image,scale=0,prob=1),image)


class RotationRangeTests(unittest.TestCase):
    def test_affine_rotation_uses_requested_range(self):
        from PIL import Image
        api=module(self,'data')
        def uniform(low,high): return 1 if (low,high)==(.9,1.1) else 10 if high==10 else 0
        with patch('random.random',side_effect=[1,1,0,1,1,1]),patch('random.uniform',side_effect=uniform) as draw:
            output=api.Augment('robust_v2',{'rotation_range':10})(Image.new('RGB',(32,32),'white'))
        self.assertEqual(draw.call_args_list[0],unittest.mock.call(-10,10))
        self.assertEqual(output.size,(32,32))

    def test_legacy_rotation_remains_five_degrees(self):
        from PIL import Image
        api=module(self,'data')
        def uniform(low,high): return 1 if (low,high)==(.9,1.1) else 0
        with patch('random.random',side_effect=[1,1,0,1]),patch('random.uniform',side_effect=uniform) as draw:
            api.Augment('basic')(Image.new('RGB',(32,32),'white'))
        self.assertEqual(draw.call_args_list[0],unittest.mock.call(-5,5))


class CropPadTests(unittest.TestCase):
    def test_crop_pad_restores_shape_with_black_padding(self):
        from PIL import Image
        image=Image.new('RGB',(32,32),'white')
        with patch('random.random',return_value=0),patch('random.randint',return_value=0):
            output=module(self,'data').random_crop_pad(image,crop_ratio=.5,prob=1)
        self.assertEqual(output.size,image.size)
        self.assertEqual(output.getpixel((0,0)),(0,0,0))
        self.assertEqual(output.getpixel((16,16)),(255,255,255))

    def test_crop_pad_can_be_disabled(self):
        image=torch.rand(3,8,8);api=module(self,'data')
        self.assertIs(api.random_crop_pad(image,prob=0),image)
        self.assertIs(api.random_crop_pad(image,crop_ratio=1,prob=1),image)


class ElasticTests(unittest.TestCase):
    def test_missing_scipy_warns_and_preserves_original(self):
        from PIL import Image
        api=module(self,'data');image=Image.new('RGB',(16,16))
        with patch('random.random',return_value=0),patch('deepfake_detection.data.datasets.importlib.import_module',side_effect=ImportError('missing')):
            with self.assertWarnsRegex(RuntimeWarning,'scipy'):
                output=api.elastic_deform(image,prob=1)
        self.assertIs(output,image)

    def test_elastic_default_is_disabled_without_import(self):
        from PIL import Image
        api=module(self,'data');image=Image.new('RGB',(16,16))
        with patch('deepfake_detection.data.datasets.importlib.import_module') as optional:
            self.assertIs(api.elastic_deform(image),image)
        optional.assert_not_called()

    def test_available_elastic_backend_preserves_dimensions(self):
        from PIL import Image
        import importlib.util
        if importlib.util.find_spec('scipy') is None: self.skipTest('optional scipy not installed')
        api=module(self,'data');image=Image.fromarray(np.random.default_rng(42).integers(0,256,(16,16,3),dtype=np.uint8))
        output=api.elastic_deform(image,alpha=2,sigma=.1,prob=1)
        self.assertEqual(output.size,image.size);self.assertEqual(output.mode,'RGB')


class BlurRangeTests(unittest.TestCase):
    def test_blur_uses_configured_sigma_range(self):
        from PIL import Image
        image=Image.new('RGB',(16,16));image.putpixel((8,8),(255,255,255))
        with patch('random.random',side_effect=[1,1,1,0,1,1]),patch('random.uniform',return_value=1.5) as sigma:
            output=module(self,'data').Augment('robust_v2',{'blur_sigma_range':[.1,1.5]})(image)
        sigma.assert_called_once_with(.1,1.5)
        self.assertLess(output.getpixel((8,8))[0],255)

    def test_gaussian_blur_can_be_disabled(self):
        from PIL import Image
        image=Image.new('RGB',(16,16));api=module(self,'data')
        self.assertIs(api.random_gaussian_blur(image,prob=0),image)
        self.assertIs(api.random_gaussian_blur(image,[0,0],prob=1),image)


class MotionBlurTests(unittest.TestCase):
    def test_motion_kernel_spreads_impulse_along_selected_direction(self):
        from PIL import Image
        image=Image.new('RGB',(17,17));image.putpixel((8,8),(255,255,255))
        with patch('random.random',return_value=0),patch('random.choice',return_value='horizontal'):
            output=module(self,'data').motion_blur(image,kernel_size=5,prob=1)
        values=np.asarray(output)[:,:,0]
        self.assertEqual(int((values[8]>0).sum()),5)
        self.assertEqual(int((values[:,8]>0).sum()),1)
        self.assertEqual(output.size,image.size)

    def test_motion_blur_zero_probability_or_unit_kernel_is_noop(self):
        image=torch.rand(3,8,8);api=module(self,'data')
        self.assertIs(api.motion_blur(image,prob=0),image)
        self.assertIs(api.motion_blur(image,kernel_size=1,prob=1),image)


class GrayscaleTests(unittest.TestCase):
    def test_grayscale_keeps_three_equal_channels(self):
        from PIL import Image
        image=Image.new('RGB',(16,16),(255,30,10))
        with patch('random.random',return_value=0):
            output=module(self,'data').random_grayscale(image,prob=1)
        values=np.asarray(output)
        self.assertEqual(output.mode,'RGB')
        self.assertTrue(np.array_equal(values[:,:,0],values[:,:,1]))
        self.assertTrue(np.array_equal(values[:,:,1],values[:,:,2]))

    def test_grayscale_probability_zero_returns_original(self):
        image=torch.rand(3,8,8)
        self.assertIs(module(self,'data').random_grayscale(image,prob=0),image)


class RandAugmentTests(unittest.TestCase):
    def test_randaugment_is_seeded_and_preserves_rgb_dimensions(self):
        from PIL import Image
        image=Image.fromarray(np.random.default_rng(42).integers(0,256,(16,16,3),dtype=np.uint8));api=module(self,'data')
        torch.manual_seed(42);first=api.randaugment(image,num_ops=2,magnitude=9)
        torch.manual_seed(42);second=api.randaugment(image,num_ops=2,magnitude=9)
        self.assertTrue(np.array_equal(np.asarray(first),np.asarray(second)))
        self.assertEqual(first.size,image.size);self.assertEqual(first.mode,'RGB')

    def test_manual_subset_is_available_without_native_policy(self):
        from PIL import Image
        with patch('torchvision.transforms.RandAugment',None):
            output=module(self,'data').randaugment(Image.new('RGB',(16,16),'red'),2,9)
        self.assertEqual(output.size,(16,16))

    def test_zero_operations_disable_and_conflicting_policies_fail(self):
        from PIL import Image
        api=module(self,'data');image=Image.new('RGB',(16,16))
        self.assertIs(api.randaugment(image,num_ops=0),image)
        with self.assertRaisesRegex(ValueError,'policy'):
            api.Augment('robust_v2',{'randaugment':True,'trivialaugment':True})


class TrivialAugmentTests(unittest.TestCase):
    def test_trivial_policy_seed_and_shape(self):
        from PIL import Image
        image=Image.fromarray(np.random.default_rng(42).integers(0,256,(16,16,3),dtype=np.uint8));api=module(self,'data')
        torch.manual_seed(42);first=api.trivialaugment(image)
        torch.manual_seed(42);second=api.trivialaugment(image)
        self.assertTrue(np.array_equal(np.asarray(first),np.asarray(second)))
        self.assertEqual(first.size,image.size)

    def test_trivial_fallback_and_disable(self):
        from PIL import Image
        api=module(self,'data');image=Image.new('RGB',(16,16),'red')
        self.assertIs(api.trivialaugment(image,enabled=False),image)
        with patch('torchvision.transforms.TrivialAugmentWide',None):
            self.assertEqual(api.trivialaugment(image).size,image.size)


class AugMixTests(unittest.TestCase):
    def test_augmix_respects_branch_and_original_mixture_weights(self):
        from PIL import Image
        image=Image.new('RGB',(16,16),'white');black=Image.new('RGB',(16,16),'black')
        with patch('numpy.random.dirichlet',return_value=np.array([.25,.75])),patch('numpy.random.beta',return_value=.5),patch('deepfake_detection.data.datasets._policy_fallback',side_effect=[black,image]):
            output=module(self,'data').augmix(image,num_aug=2,depth=1,alpha=1)
        self.assertEqual(output.getpixel((8,8)),(223,223,223))

    def test_augmix_seeded_and_disableable(self):
        import random
        from PIL import Image
        api=module(self,'data');image=Image.new('RGB',(16,16),'red')
        self.assertIs(api.augmix(image,enabled=False),image)
        np.random.seed(42);random.seed(42);first=api.augmix(image)
        np.random.seed(42);random.seed(42);second=api.augmix(image)
        self.assertTrue(np.array_equal(np.asarray(first),np.asarray(second)))


class StyleTransferTests(unittest.TestCase):
    def test_style_transfer_stub_reports_external_requirements(self):
        from PIL import Image
        with self.assertRaisesRegex(NotImplementedError,'style images.*weights'):
            module(self,'data').style_transfer(Image.new('RGB',(16,16)))

    def test_disabled_style_transfer_is_noop(self):
        from PIL import Image
        image=Image.new('RGB',(16,16))
        self.assertIs(module(self,'data').style_transfer(image,enabled=False),image)


class ArtifactInjectionTests(unittest.TestCase):
    def test_dataset_pseudo_fake_and_preserve_policies_are_explicit(self):
        from tests.fixtures import crop_fixture
        ws,cfg,rows=crop_fixture(self,splits=('train',))
        api=module(self,'data')
        cfg['training'].update(artifact_injection=True,artifact_prob=1,artifact_label_policy='pseudo_fake')
        def changed(image,prob):
            result=image.copy();result.putpixel((0,0),(255,0,0));return result
        with patch('deepfake_detection.data.datasets.inject_blend_artifact',side_effect=changed),patch('deepfake_detection.data.datasets.inject_color_mismatch',side_effect=lambda image,prob:image),patch('deepfake_detection.data.datasets.inject_warp_artifact',side_effect=lambda image,prob:image):
            dataset=api.CropDataset(rows,cfg,training=True,workspace=ws)
            self.assertEqual(dataset[1][1],0)
            self.assertEqual(dataset.timed_item(1)[1],0)
            cfg['training']['artifact_label_policy']='preserve'
            self.assertEqual(api.CropDataset(rows,cfg,training=True,workspace=ws)[1][1],1)
            self.assertEqual(api.CropDataset(rows,cfg,training=False,workspace=ws)[1][1],1)

    def test_enabled_artifacts_require_label_policy(self):
        from tests.fixtures import crop_fixture
        ws,cfg,rows=crop_fixture(self,splits=('train',))
        cfg['training']['artifact_injection']=True
        with self.assertRaisesRegex(ValueError,'artifact_label_policy'):
            module(self,'data').CropDataset(rows,cfg,training=True,workspace=ws)[1]
    def image(self):
        from PIL import Image
        return Image.fromarray(np.random.default_rng(42).integers(0,256,(32,32,3),dtype=np.uint8))
    def test_blend_artifact_changes_bounded_pixels(self):
        image=self.image()
        output=module(self,'data').inject_blend_artifact(image,prob=1)
        self.assertEqual(output.size,image.size)
        changed=(np.asarray(output)!=np.asarray(image)).any(axis=2)
        self.assertTrue(changed.any());self.assertFalse(changed.all())
    def test_color_mismatch_changes_bounded_pixels(self):
        image=self.image()
        output=module(self,'data').inject_color_mismatch(image,prob=1)
        self.assertEqual(output.size,image.size)
        changed=(np.asarray(output)!=np.asarray(image)).any(axis=2)
        self.assertTrue(changed.any());self.assertFalse(changed.all())
    def test_local_warp_preserves_shape_and_finite_pixels(self):
        image=self.image()
        output=module(self,'data').inject_warp_artifact(image,prob=1)
        self.assertEqual(output.size,image.size)
        self.assertFalse(np.array_equal(np.asarray(output),np.asarray(image)))
    def test_artifact_functions_disabled_are_exact_noops(self):
        api=module(self,'data');image=self.image()
        for function in (api.inject_blend_artifact,api.inject_color_mismatch,api.inject_warp_artifact):
            self.assertIs(function(image,prob=0),image)
