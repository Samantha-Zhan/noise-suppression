classdef permuteLayer < nnet.layer.Layer & nnet.layer.Formattable
    properties
        isSpecial
        newDimension
    end
    properties (Learnable)
    end
    methods
        function layer = permuteLayer(isSpecial, newDimension, NVargs)
            arguments
                isSpecial = false
                newDimension = []
                NVargs.Name string = "reshape"
            end
            layer.isSpecial = isSpecial;
            layer.newDimension = newDimension;
            layer.Name = NVargs.Name;
        end
        function Z = predict(layer, X)
            formated = permute(X, layer.newDimension);
            if(layer.isSpecial)
                formated = reshape(formated,[],257,shape(formated,2),shape(formated,3));
                formated = permute(formated,[1,3,2,4]);
                formated = formated(:,:,:,2:end);
            end
            Z = formated;
        end
    end
    
end