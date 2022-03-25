classdef unfoldLayer < nnet.layer.Layer & nnet.layer.Formattable
    properties
    end
    properties (Learnable)
    end
    methods
        function layer = unfoldLayer(NVargs)
            arguments
                NVargs.Name string = "unfoldLayer"
            end
            layer.Name = NVargs.Name;
        end
        function Z = predict(layer, X)
            num_neighbors=15;
            % !
            [batch_size, num_channels, num_freqs, num_frames] = size(X);
            output = X.reshape(batch_size * num_channels, 1, num_freqs, num_frames);
            sub_band_unit_size = num_neighbors * 2 + 1;
            % !
            output = padarray(output,[0, 0, num_neighbors, 0],'symmetric','both');
            output = im2col(output, [sub_band_unit_size, num_frames]);
            output = reshape(output, [batch_size, num_channels, sub_band_unit_size, num_frames, num_freqs]);
            output = permute(output,[0,4,1,2,3]);
            output = reshape(output, [batch_size, num_freqs, sub_band_unit_size, num_frames]);
            Z = output;
        end
    end
    
end