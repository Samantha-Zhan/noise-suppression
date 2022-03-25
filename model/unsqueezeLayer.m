classdef unsqueezeLayer < nnet.layer.Layer & nnet.layer.Formattable
    properties
        indexNum
    end
    properties (Learnable)
    end
    methods
        function layer = unsqueezeLayer(indexNum, NVargs)
            arguments
                indexNum = 2
                NVargs.Name string = "pad"
            end
            layer.indexNum = indexNum;
            layer.Name = NVargs.Name;
        end
        function Z = predict(layer, X)
            formated = X;
            switch layer.indexNum
               case 1
                  formated = reshape(1,size(X,1), size(X,2),size(X,3));
               case 2
                  formated = reshape(size(X,1), 1,size(X,2),size(X,3));
               case 3
                  formated = reshape(size(X,1), size(X,2),1,size(X,3));
                case 4
                  formated = reshape(size(X,1), size(X,2),size(X,3),1);
               otherwise
                  formated = X;
            end
            Z = formated;
        end
    end
    
end