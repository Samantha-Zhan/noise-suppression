classdef padLayer < nnet.layer.Layer & nnet.layer.Formattable
    properties
    end
    properties (Learnable)
    end
    methods
        function layer = padLayer(NVargs)
            arguments
                NVargs.Name string = "pad"
            end
            layer.Name = NVargs.Name;
        end
        function Z = predict(layer, X)
            % formated = reshape(size(X,1), 1,size(X,2),size(X,3));
            formated = padarray(X, [0,0,0,2],0,'post');
            Z = formated;
        end
    end
    
end