classdef normLayer < nnet.layer.Layer & nnet.layer.Formattable
    properties
        isSpecial
    end
    properties (Learnable)
    end
    methods
        function layer = normLayer(isSpecial, NVargs)
            arguments
                isSpecial = false
                NVargs.Name string = "norm"
            end
            layer.isSpecial = isSpecial;
            layer.Name = NVargs.Name;
        end
        function Z = predict(layer, X)
            formated = X;
            % mean normalization
            mu = mean(formated,'all');
            normed = formated / (mu + 1e-5);
            if(layer.isSpecial)
                normed = reshape(normed, [shape(normed,1)*shape(normed,2),shape(normed,3),shape(normed,4)]);
                normed = permute(normed, [1,3,2]);
            end
            % max normalization
            % maxVal = max( X ,[], 'all' )+ 1e-5;
            % normed = X / maxVal;
            Z = normed;
        end
    end
    
end